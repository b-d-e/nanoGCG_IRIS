"""
Script to run GCG attacks on a dataset across multiple GPUs.
"""

import argparse
import os

# Set HF cache dir for the main process
hf_cache_dir = "/scratch/etheridge/huggingface/"
os.environ["HF_HOME"] = hf_cache_dir
print(f"Setting HuggingFace cache directory to {hf_cache_dir}")


import time
import gc
import sys
from datetime import datetime
from multiprocessing import Process, Queue
import pandas as pd
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import nanoGCG
import nanogcg
from nanogcg import GCGConfig, ProbeSamplingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GCG attacks on a dataset.")
    parser.add_argument("--input-csv", type=str, default="../orthogonalized_outputs_2048.csv",
                        help="Path to the input CSV file.")
    parser.add_argument("--output-csv", type=str, default="gcg_attacks_results.csv",
                        help="Path to save the output CSV file.")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Victim model to use.")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Data type for model loading.")
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="Number of GPUs to use.")
    parser.add_argument("--model-memory-upper-bound-gb", type=int, default=30, # generous estimate
                        help="Estimated memory usage per model in GB.")
    parser.add_argument("--gpu-memory-lower-bound-gb", type=int, default=90-18,
                        help="Minimum available memory per GPU in GB.")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens for generation.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to process (None for all).")
    parser.add_argument("--wandb-log", action="store_true", default=True,
                        help="Whether to log to W&B.")
    parser.add_argument("--wandb-entity", type=str, default="reasoning_attacks",
                        help="W&B entity.")
    parser.add_argument("--wandb-project", type=str, default="refusal_scores",
                        help="W&B project.")
    parser.add_argument("--probe-sampling", action="store_true",
                        help="Use probe sampling for GCG.")
    parser.add_argument("--early-stop", action="store_true",
                        help="Early stop GCG if a perfect match is found.")
    parser.add_argument("--num-steps", type=int, default=250,
                        help="Number of GCG optimization steps.")
    parser.add_argument("--hf-cache-dir", type=str, default="/scratch/etheridge/huggingface/",
                        help="HuggingFace cache directory.")

    return parser.parse_args()


def run_attack(gpu_id, job_queue, result_queue, args):
    """
    Run GCG attacks on a specific GPU.
    
    Args:
        gpu_id: GPU ID to use
        job_queue: Queue of jobs to process
        result_queue: Queue to store results
        args: Command line arguments
    """
    # Set the device
    device = f"cuda:{gpu_id}"
    print(f"Worker {gpu_id} starting on device {device}")
    
    # Set HF cache dir
    os.environ["HF_HOME"] = args.hf_cache_dir
    
    # Load the model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=getattr(torch, args.dtype),
            trust_remote_code=True,
            device_map=device
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # Set up probe sampling if requested
        probe_sampling_config = None
        if args.probe_sampling:
            # Load a smaller model as draft model
            draft_model = AutoModelForCausalLM.from_pretrained(
                "openai-community/gpt2", 
                torch_dtype=getattr(torch, args.dtype)
            ).to(device)
            draft_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
            probe_sampling_config = ProbeSamplingConfig(
                draft_model=draft_model,
                draft_tokenizer=draft_tokenizer,
            )
        
        # Main processing loop
        while True:
            job = job_queue.get()
            if job is None:  # Sentinel value to indicate end of queue
                break
                
            idx, row = job
            print(f"Worker {gpu_id} processing row {idx}...")
            start_time = time.time()
            
            try:
                # Create configuration for GCG
                wandb_config = None
                if args.wandb_log:
                    # Create a unique run name based on row index and first part of prompt
                    run_name = f"row_{idx}_{row['forbidden_prompt'].split('.')[0]}"
                    wandb_config = {
                        "entity": args.wandb_entity,
                        "project": args.wandb_project,
                        "name": run_name,
                    }
                
                config = GCGConfig(
                    verbosity="INFO",
                    probe_sampling_config=probe_sampling_config,
                    early_stop=args.early_stop,
                    num_steps=args.num_steps,
                    wandb_config=wandb_config,
                    seed=42,
                )
                
                # Run GCG
                messages = [{"role": "user", "content": row['forbidden_prompt']}]
                result = nanogcg.run(
                    model,
                    tokenizer,
                    messages,
                    row['gcg_target'],
                    config,
                )
                
                # Get the optimized string
                best_string = result.best_string
                best_loss = result.best_loss
                
                # Generate with the model using the optimized prompt
                full_prompt = row['forbidden_prompt'] + " " + best_string
                messages = [{"role": "user", "content": full_prompt}]
                input_tensor = tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    return_tensors="pt"
                ).to(device)
                
                output = model.generate(
                    input_tensor, 
                    do_sample=False, 
                    max_new_tokens=args.max_tokens
                )
                
                response = tokenizer.batch_decode(
                    output[:, input_tensor.shape[1]:], 
                    skip_special_tokens=True
                )[0]
                
                elapsed_time = time.time() - start_time
                print(f"Worker {gpu_id} finished row {idx} in {elapsed_time:.2f}s")
                
                # Send results back
                result_queue.put((idx, {
                    'gcg_best_string': best_string,
                    'gcg_best_loss': best_loss,
                    'gcg_response': response,
                    'gcg_execution_time': elapsed_time
                }))
                
            except Exception as e:
                print(f"Worker {gpu_id} error processing row {idx}: {e}")
                result_queue.put((idx, {
                    'gcg_best_string': f"ERROR: {str(e)}",
                    'gcg_best_loss': -1,
                    'gcg_response': f"ERROR: {str(e)}",
                    'gcg_execution_time': time.time() - start_time
                }))
            
            # Clean up
            gc.collect()
            torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Worker {gpu_id} failed to initialize: {e}")
        # Put None in the result queue to signal this worker is done
        result_queue.put(None)
    
    # Signal that this worker is done
    result_queue.put(None)
    print(f"Worker {gpu_id} finished")


def main():
    """Main function."""
    args = parse_args()
    
    # Calculate number of attacks per GPU
    attacks_per_gpu = args.gpu_memory_lower_bound_gb // args.model_memory_upper_bound_gb
    print(f"Will run {attacks_per_gpu} attacks per GPU on {args.num_gpus} GPUs")


    # Load the data
    print(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    # Limit to number of samples if specified
    if args.num_samples is not None:
        df = df.head(args.num_samples)
    
    # Add GCG target column - first two sentences of 'response'
    # df['gcg_target'] = df['response'].apply(
    #     lambda x: x.split(". ")[0] + ". " + x.split(". ")[1] + "."
    # )
    # Add GCG target - "Sure, here is" to all rows
    df['gcg_target'] = "Sure, here is"
    
    # Add result columns
    df['gcg_best_string'] = ""
    df['gcg_best_loss'] = None
    df['gcg_response'] = ""
    df['gcg_execution_time'] = None
    
    # if output csv doesn't exist, create it
    if not os.path.exists(args.output_csv):
        df.to_csv(args.output_csv, index=False)
    else:
        print(f"Output CSV {args.output_csv} already exists")

    # Print data info
    print(f"Loaded {len(df)} rows from {args.input_csv}")

    print(df.head())
    
    # Create queues for job distribution and result collection
    job_queue = Queue()
    result_queue = Queue()
    
    # Add all jobs to the queue
    for idx, row in df.iterrows():
        job_queue.put((idx, row))
    
    # Add sentinel values to signal workers to stop
    for _ in range(args.num_gpus * attacks_per_gpu):
        job_queue.put(None)
    
    # Start worker processes
    processes = []
    for gpu_id in range(args.num_gpus):
        for i in range(attacks_per_gpu):
            p = Process(
                target=run_attack, 
                args=(gpu_id, job_queue, result_queue, args)
            )
            processes.append(p)
            p.start()
    
    # Collect results
    num_completed = 0
    num_workers = args.num_gpus * attacks_per_gpu
    active_workers = num_workers
    
    print(f"Started {num_workers} workers, waiting for results...")
    
    while active_workers > 0:
        result = result_queue.get()
        if result is None:
            active_workers -= 1
            continue
            
        idx, data = result
        num_completed += 1
        
        # Update DataFrame
        for key, value in data.items():
            df.at[idx, key] = value
        
        # Periodically save progress
        # if num_completed % 10 == 0 or active_workers == 0:
        if num_completed % num_workers == 0 or active_workers == 0:
            print(f"Completed {num_completed}/{len(df)} jobs. Saving progress...")
            df.to_csv(args.output_csv, index=False)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Final save
    df.to_csv(args.output_csv, index=False)
    print(f"All done! Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()