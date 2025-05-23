import copy
import gc
import logging
import queue
import threading
import signal
import sys
import atexit

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Tuple, Union, Dict, Any
from typing import Callable
import torch
import transformers
from torch import Tensor
from transformers import set_seed
from scipy.stats import spearmanr
import wandb
import numpy as np

from nanogcg.utils import (
    INIT_CHARS,
    configure_pad_token,
    find_executable_batch_size,
    get_nonascii_toks,
    mellowmax,
)

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ProbeSamplingConfig:
    draft_model: transformers.PreTrainedModel
    draft_tokenizer: transformers.PreTrainedTokenizer
    r: int = 8
    sampling_factor: int = 16

@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int  = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    probe_sampling_config: Optional[ProbeSamplingConfig] = None
    wandb_config: Optional[dict] = None
    run_id: Optional[str] = None 
    prompt_string: Optional[str] = None
    target_string: Optional[str] = None
    target_no_think: Optional[bool] = None

    # refusal direction minimisation terms
    use_refusal_direction: bool = False
    refusal_vector_path: Optional[str] = None
    refusal_vector: Optional[torch.Tensor] = None
    refusal_layer_idx: int = None 
    refusal_num_tokens: int = None
    refusal_beta: float = 0.75  # weight balancing term
    
    # New option to use the final prompt token
    use_final_prompt_token: bool = False  # If True, use final prompt token instead of first response tokens
    
    # New scheduling parameters
    use_refusal_beta_schedule: bool = False
    schedule_type: str = "linear"  # "linear" or "cosine"
    refusal_beta_start: float = 0.0
    refusal_beta_end: float = 0.75
    schedule_start_step: int = 0
    schedule_end_step: Optional[int] = None  # If None, will use num_steps
    schedule_exponent: float = 2.0  # For exponential schedule
    schedule_power: float = 2.0     # For power schedule

@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]
    best_answer: Optional[str] = None


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)


def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace : int
            the number of token positions to update per sequence
        not_allowed_ids : Tensor, shape = (n_ids)
            the token ids that should not be used in optimization

    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids)
            token ids
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer

    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
            filtered_ids.append(ids[i])

    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )

    return torch.stack(filtered_ids)


class GCG:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=model.device)
        self.prefix_cache = None
        self.draft_prefix_cache = None

        self.stop_flag = False
        self.step = 0  # Initialize step counter
        self.wandb_metrics = {}  # Initialize metrics dictionary

        self.draft_model = None
        self.draft_tokenizer = None
        self.draft_embedding_layer = None
        if self.config.probe_sampling_config:
            self.draft_model = self.config.probe_sampling_config.draft_model
            self.draft_tokenizer = self.config.probe_sampling_config.draft_tokenizer
            self.draft_embedding_layer = self.draft_model.get_input_embeddings()
            if self.draft_tokenizer.pad_token is None:
                configure_pad_token(self.draft_tokenizer)

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        if not tokenizer.chat_template:
            logger.warning("Tokenizer does not have a chat template. Assuming base model and setting chat template to empty.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


        # Initialize refusal direction if enabled
        if config.use_refusal_direction:
            if config.refusal_vector is not None:
                self.refusal_vector = config.refusal_vector.to(model.device, model.dtype)
                logger.info(f"Using provided refusal vector with shape {self.refusal_vector.shape}")
            elif config.refusal_vector_path:
                try:
                    # Try loading with torch
                    self.refusal_vector = torch.load(config.refusal_vector_path, map_location=model.device)
                    logger.info(f"Loaded refusal vector from {config.refusal_vector_path} using torch.load")
                except:
                    # Fall back to numpy
                    try:
                        refusal_array = np.load(config.refusal_vector_path)
                        self.refusal_vector = torch.tensor(refusal_array, device=model.device)
                        logger.info(f"Loaded refusal vector from {config.refusal_vector_path} using numpy")
                    except Exception as e:
                        logger.error(f"Failed to load refusal vector: {e}")
                        raise
                
                # Convert to model's data type
                self.refusal_vector = self.refusal_vector.to(dtype=model.dtype)
                logger.info(f"Loaded refusal vector with shape {self.refusal_vector.shape} and dtype {self.refusal_vector.dtype}")
                
            else:
                raise ValueError("When use_refusal_direction=True, must provide either refusal_vector or refusal_vector_path")
            
            # Normalize the refusal vector for stable dot product calculations
            self.refusal_vector = self.refusal_vector / torch.norm(self.refusal_vector)
            
            # Register hooks for activation extraction
            self.register_activation_hooks()
            logger.info("Registered activation hooks for refusal direction")

        if config.wandb_config:
            # if old notebook wandb run found, clear it
            if wandb.run is not None:
                wandb.finish()
                logger.info("Cleared old wandb run.")

            self.using_wandb = True

            wandb.init(
                project=config.wandb_config["project"],
                entity=config.wandb_config["entity"],
                config=config,
            )
            wandb.run.name = config.wandb_config["name"]

            logger.info(f"Initialized wandb run: {wandb.run.name}")

        else:
            self.using_wandb = False
            logger.info("Wandb not initialized.")

        atexit.register(self._cleanup)

    def _cleanup(self):
        if self.using_wandb:
            wandb.finish()
            logger.info("Wandb run finished.")
    
    def add_wandb_metric(self, name: str, value: Any) -> None:
        """Add a metric to the wandb metrics dictionary."""
        if self.using_wandb:
            self.wandb_metrics[name] = value
    
    def log_wandb_metrics(self) -> None:
        """Log all collected metrics to wandb."""
        if self.using_wandb and self.wandb_metrics:
            # Add step to metrics
            self.wandb_metrics['step'] = self.step
            
            # Log all metrics at once
            wandb.log(self.wandb_metrics)
            
            # Clear metrics dictionary for next step
            self.wandb_metrics = {}
    
    def run(
        self,
        messages: Union[str, List[dict]],
        target: str,
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)

        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")

        target = " " + target if config.add_space_before_target else target

        # Tokenize everything that doesn't get optimized
        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        # Store the prompt length for refusal direction calculation
        # The prompt consists of before_ids + optim_ids + after_ids
        # We'll calculate the exact position during loss computation
        if config.use_final_prompt_token:
            # Store the length up to where the prompt ends (before target begins)
            # This will be used to identify the final prompt token position
            self.prompt_length = before_ids.shape[1] + len(tokenizer(config.optim_str_init, add_special_tokens=False)["input_ids"]) + after_ids.shape[1]
        
        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=before_embeds, use_cache=True)
                self.prefix_cache = output.past_key_values

        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds

        # Initialize components for probe sampling, if enabled.
        if config.probe_sampling_config:
            assert self.draft_model and self.draft_tokenizer and self.draft_embedding_layer, "Draft model wasn't properly set up."

            # Tokenize everything that doesn't get optimized for the draft model
            draft_before_ids = self.draft_tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            draft_after_ids = self.draft_tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            self.draft_target_ids = self.draft_tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

            (
                self.draft_before_embeds,
                self.draft_after_embeds,
                self.draft_target_embeds,
            ) = [
                self.draft_embedding_layer(ids)
                for ids in (
                    draft_before_ids,
                    draft_after_ids,
                    self.draft_target_ids,
                )
            ]

            if config.use_prefix_cache:
                with torch.no_grad():
                    output = self.draft_model(inputs_embeds=self.draft_before_embeds, use_cache=True)
                    self.draft_prefix_cache = output.past_key_values

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []

        for step in tqdm(range(config.num_steps)):
            self.step = step  # Update step counter
            
            # Compute the token gradient
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():
                # Sample candidate token sequences based on the token gradient
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0]

                # Store gradient statistics for wandb
                if self.using_wandb:
                    self.add_wandb_metric("search_width", new_search_width)
                    self.add_wandb_metric("gradient_min", optim_ids_onehot_grad.min().item())

                # Compute loss on all candidate sequences
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                if self.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                else:
                    input_embeds = torch.cat([
                        before_embeds.repeat(new_search_width, 1, 1),
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)

                if self.config.probe_sampling_config is None:
                    loss = find_executable_batch_size(self._compute_candidates_loss_original, batch_size)(input_embeds)
                    current_loss = loss.min().item()
                    optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)
                else:
                    current_loss, optim_ids = find_executable_batch_size(self._compute_candidates_loss_probe_sampling, batch_size)(
                        input_embeds, sampled_ids,
                    )

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(tokenizer)

            # Log current optimization state to wandb
            if self.using_wandb:
                # # Calculate refusal term for the current best candidate
                # if self.config.use_refusal_direction and hasattr(self, 'activations') and 'target_layer' in self.activations:
                #     token_activations = self.activations['target_layer'][:, :self.config.refusal_num_tokens]
                #     mean_activations = torch.mean(token_activations, dim=1)
                #     norm_activations = mean_activations / (torch.norm(mean_activations, dim=1, keepdim=True) + 1e-6)
                #     refusal_alignments = torch.matmul(norm_activations, self.refusal_vector)
                    
                #     self.add_wandb_metric("refusal_alignment_mean", refusal_alignments.mean().item())
                #     self.add_wandb_metric("refusal_alignment_std", refusal_alignments.std().item())
                #     self.add_wandb_metric("refusal_alignment_min", refusal_alignments.min().item())

                # Add loss metrics
                best_loss = buffer.get_lowest_loss()
                self.add_wandb_metric("loss", current_loss)
                self.add_wandb_metric("loss_delta", losses[-2] - current_loss if len(losses) > 1 else 0)
                
                # Add probe sampling metrics if using
                if self.config.probe_sampling_config is not None and hasattr(self, 'last_rank_correlation'):
                    self.add_wandb_metric("rank_correlation", self.last_rank_correlation)
                    self.add_wandb_metric("alpha", (1 + self.last_rank_correlation) / 2)

                # Log all metrics at once
                self.log_wandb_metrics()
                
                # Update summary with latest best string
                best_string_value = tokenizer.batch_decode(buffer.get_best_ids())[0]
                wandb.run.summary["best_string"] = best_string_value
                wandb.run.summary["best_loss"] = best_loss

                # Generate and log model response periodically
                if step % 25 == 0:
                    full_prompt = self.config.prompt_string + " " + best_string_value
                    messages = [{"role": "user", "content": full_prompt}]
                    input_tensor = tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt=True, 
                        return_tensors="pt"
                    ).to(model.device, torch.int64)
                    output = model.generate( 
                        input_tensor, 
                        do_sample=False, 
                        max_new_tokens=2048,
                    )
                    response = tokenizer.batch_decode(
                        output[:, input_tensor.shape[1]:], 
                        skip_special_tokens=True
                    )[0]
                    wandb.run.summary["best_answer"] = response

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                if self.using_wandb:
                    wandb.run.summary["early_stopped"] = True
                break

        min_loss_index = losses.index(min(losses))

        # Generate response and log to wandb
        best_string_value = tokenizer.batch_decode(buffer.get_best_ids())[0]
        full_prompt = self.config.prompt_string + " " + best_string_value
        print(full_prompt)

        messages = [{"role": "user", "content": full_prompt}]
        input_tensor = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(model.device, torch.int64)
        
        output = model.generate(
            input_tensor, 
            do_sample=False, 
            max_new_tokens=2048
        )
        
        response = tokenizer.batch_decode(
            output[:, input_tensor.shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        if self.using_wandb:
            wandb.run.summary["best_answer"] = response

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
            best_answer=response,
        )

        wandb.finish()

        return result

    def init_buffer(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
                init_indices = torch.randint(0, init_buffer_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1]))
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids

        else:  # assume list
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_buffer_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            except ValueError:
                logger.error("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        true_buffer_size = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
        else:
            init_buffer_embeds = torch.cat([
                self.before_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)

        init_buffer_losses = find_executable_batch_size(self._compute_candidates_loss_original, true_buffer_size)(init_buffer_embeds)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer)

        logger.info("Initialized attack buffer.")

        return buffer

    def compute_token_gradient(self, optim_ids: Tensor) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix."""
        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        if self.prefix_cache:
            input_embeds = torch.cat([optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            output = model(
                inputs_embeds=input_embeds,
                past_key_values=self.prefix_cache,
                use_cache=True,
            )
        else:
            input_embeds = torch.cat(
                [
                    self.before_embeds,
                    optim_embeds,
                    self.after_embeds,
                    self.target_embeds,
                ],
                dim=1,
            )
            output = model(inputs_embeds=input_embeds)

        logits = output.logits

        # Shift logits so token n-1 predicts token n
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift - 1 : -1, :].contiguous()  # (1, num_target_ids, vocab_size)
        shift_labels = self.target_ids

        # Use the new loss function
        loss = self.compute_loss_with_refusal_direction(shift_logits, shift_labels)

        optim_ids_onehot_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

        return optim_ids_onehot_grad

    def _compute_candidates_loss_original(self, search_batch_size: int, input_embeds: Tensor) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences."""
        all_loss = []
        prefix_cache_batch = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if self.prefix_cache:
                    if not prefix_cache_batch or current_batch_size != search_batch_size:
                        prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[i]] for i in range(len(self.prefix_cache))]

                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch, use_cache=True)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits

                tmp = input_embeds.shape[1] - self.target_ids.shape[1]
                shift_logits = logits
                tmp = input_embeds.shape[1] - self.target_ids.shape[1]
                shift_logits = logits[..., tmp-1:-1, :].contiguous()
                shift_labels = self.target_ids.repeat(current_batch_size, 1)

                # Use the new loss function
                batch_loss = self.compute_loss_with_refusal_direction(shift_logits, shift_labels)
                all_loss.append(batch_loss)

                if self.config.early_stop:
                    if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
                        self.stop_flag = True

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

    def _compute_candidates_loss_probe_sampling(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
        sampled_ids: Tensor,
    ) -> Tuple[float, Tensor]:
        """Computes the GCG loss using probe sampling (https://arxiv.org/abs/2403.01251).

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
            sampled_ids: Tensor, all candidate token id sequences. Used for further sampling.

        Returns:
            A tuple of (min_loss: float, corresponding_sequence: Tensor)

        """

        raise NotImplementedError("Probe sampling is not tested for refusal direction optimisation yet")

        probe_sampling_config = self.config.probe_sampling_config
        assert probe_sampling_config, "Probe sampling config wasn't set up properly."

        B = input_embeds.shape[0]
        probe_size = B // probe_sampling_config.sampling_factor
        probe_idxs = torch.randperm(B)[:probe_size].to(input_embeds.device)
        probe_embeds = input_embeds[probe_idxs]

        def _compute_probe_losses(result_queue: queue.Queue, search_batch_size: int, probe_embeds: Tensor) -> None:
            probe_losses = self._compute_candidates_loss_original(search_batch_size, probe_embeds)
            result_queue.put(("probe", probe_losses))

        def _compute_draft_losses(
            result_queue: queue.Queue,
            search_batch_size: int,
            draft_sampled_ids: Tensor,
        ) -> None:
            assert self.draft_model and self.draft_embedding_layer, "Draft model and embedding layer weren't initialized properly."

            draft_losses = []
            draft_prefix_cache_batch = None
            for i in range(0, B, search_batch_size):
                with torch.no_grad():
                    batch_size = min(search_batch_size, B - i)
                    draft_sampled_ids_batch = draft_sampled_ids[i : i + batch_size]

                    if self.draft_prefix_cache:
                        if not draft_prefix_cache_batch or batch_size != search_batch_size:
                            draft_prefix_cache_batch = [
                                [x.expand(batch_size, -1, -1, -1) for x in self.draft_prefix_cache[i]] for i in range(len(self.draft_prefix_cache))
                            ]
                        draft_embeds = torch.cat(
                            [
                                self.draft_embedding_layer(draft_sampled_ids_batch),
                                self.draft_after_embeds.repeat(batch_size, 1, 1),
                                self.draft_target_embeds.repeat(batch_size, 1, 1),
                            ],
                            dim=1,
                        )
                        draft_output = self.draft_model(
                            inputs_embeds=draft_embeds,
                            past_key_values=draft_prefix_cache_batch,
                        )
                    else:
                        draft_embeds = torch.cat(
                            [
                                self.draft_before_embeds.repeat(batch_size, 1, 1),
                                self.draft_embedding_layer(draft_sampled_ids_batch),
                                self.draft_after_embeds.repeat(batch_size, 1, 1),
                                self.draft_target_embeds.repeat(batch_size, 1, 1),
                            ],
                            dim=1,
                        )
                        draft_output = self.draft_model(inputs_embeds=draft_embeds)

                    draft_logits = draft_output.logits
                    tmp = draft_embeds.shape[1] - self.draft_target_ids.shape[1]
                    shift_logits = draft_logits[..., tmp - 1 : -1, :].contiguous()
                    shift_labels = self.draft_target_ids.repeat(batch_size, 1)

                    if self.config.use_mellowmax:
                        label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                        loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                    else:
                        loss = (
                            torch.nn.functional.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                reduction="none",
                            )
                            .view(batch_size, -1)
                            .mean(dim=-1)
                        )

                    draft_losses.append(loss)

            draft_losses = torch.cat(draft_losses)
            result_queue.put(("draft", draft_losses))

        def _convert_to_draft_tokens(token_ids: Tensor) -> Tensor:
            decoded_text_list = self.tokenizer.batch_decode(token_ids)
            assert self.draft_tokenizer, "Draft tokenizer wasn't properly initialized."
            return self.draft_tokenizer(
                decoded_text_list,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )[
                "input_ids"
            ].to(self.draft_model.device, torch.int64)

        result_queue = queue.Queue()
        draft_sampled_ids = _convert_to_draft_tokens(sampled_ids)

        # Step 1. Compute loss of all candidates using the draft model
        draft_thread = threading.Thread(
            target=_compute_draft_losses,
            args=(result_queue, search_batch_size, draft_sampled_ids),
        )

        # Step 2. In parallel to 1., compute loss of the probe set on target model
        probe_thread = threading.Thread(
            target=_compute_probe_losses,
            args=(result_queue, search_batch_size, probe_embeds),
        )

        draft_thread.start()
        probe_thread.start()

        draft_thread.join()
        probe_thread.join()

        results = {}
        while not result_queue.empty():
            key, value = result_queue.get()
            results[key] = value

        probe_losses = results["probe"]
        draft_losses = results["draft"]

        # Step 3. Calculate agreement score using Spearman correlation
        draft_probe_losses = draft_losses[probe_idxs]
        rank_correlation = spearmanr(
            probe_losses.cpu().type(torch.float32).numpy(),
            draft_probe_losses.cpu().type(torch.float32).numpy(),
        ).correlation
        # normalized from [-1, 1] to [0, 1]
        alpha = (1 + rank_correlation) / 2

        self.last_rank_correlation = rank_correlation

        # Step 4. Calculate the filtered set and evaluate using the target model.
        R = probe_sampling_config.r
        filtered_size = int((1 - alpha) * B / R)
        filtered_size = max(1, min(filtered_size, B))

        _, top_indices = torch.topk(draft_losses, k=filtered_size, largest=False)

        filtered_embeds = input_embeds[top_indices]
        filtered_losses = self._compute_candidates_loss_original(search_batch_size, filtered_embeds)

        # Step 5. Return best loss between probe set and filtered set
        best_probe_loss = probe_losses.min().item()
        best_filtered_loss = filtered_losses.min().item()

        # Add metrics for wandb
        if self.using_wandb:
            self.add_wandb_metric("probe_sampling/rank_correlation", rank_correlation)
            self.add_wandb_metric("probe_sampling/alpha", alpha)
            self.add_wandb_metric("probe_sampling/probe_size", probe_size)
            self.add_wandb_metric("probe_sampling/filtered_size", filtered_size)
            self.add_wandb_metric("probe_sampling/best_probe_loss", best_probe_loss)
            self.add_wandb_metric("probe_sampling/mean_probe_loss", probe_losses.mean().item())
            self.add_wandb_metric("probe_sampling/mean_draft_loss", draft_losses.mean().item())
            self.add_wandb_metric("probe_sampling/best_filtered_loss", best_filtered_loss)

        probe_ids = sampled_ids[probe_idxs]
        filtered_ids = sampled_ids[top_indices]
        
        return (
            (best_probe_loss, probe_ids[probe_losses.argmin()].unsqueeze(0))
            if best_probe_loss < best_filtered_loss
            else (
                best_filtered_loss,
                filtered_ids[filtered_losses.argmin()].unsqueeze(0),
            )
        )

    def register_activation_hooks(self):
        """Register hooks to extract activations from specified layers."""
        self.activations = {}
        
        def get_activation(name):
            def hook(module, input, output):
                # For DeepSeek models, input[0] contains the residual stream
                self.activations[name] = input[0]
            return hook
        
        # Register hook on the desired layer's input_layernorm
        if self.config.use_refusal_direction:
            if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
                # Fallback for different model architectures
                target_layers = list(self.model.modules())
                # Find transformer layers (this might need adjustment based on model architecture)
                transformer_layers = [m for m in target_layers if "TransformerBlock" in str(type(m)) or "DecoderLayer" in str(type(m))]
                if self.config.refusal_layer_idx < len(transformer_layers):
                    target_layer = transformer_layers[self.config.refusal_layer_idx]
                    # Try to find input normalization layer
                    for name, module in target_layer.named_modules():
                        if "input_layernorm" in name or "input_norm" in name or "ln_1" in name:
                            module.register_forward_hook(get_activation('target_layer'))
                            break
            else:
                # For DeepSeek models
                try:
                    target_layer = self.model.model.layers[self.config.refusal_layer_idx].input_layernorm
                    target_layer.register_forward_hook(get_activation('target_layer'))
                except (IndexError, AttributeError) as e:
                    logger.error(f"Failed to register hook: {e}")
                    logger.error("Check if the model architecture is compatible and refusal_layer_idx is valid")
                    raise


    def compute_loss_with_refusal_direction(self, shift_logits, shift_labels):
        """Compute loss with optional refusal direction component using scheduled beta."""
        
        # Original loss calculation (cross-entropy or mellowmax)
        if self.config.use_mellowmax:
            label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            original_loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
        else:
            original_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none"
            ).view(shift_logits.shape[0], -1).mean(dim=-1)
        
        # Store original loss metrics for wandb
        if self.using_wandb:
            self.add_wandb_metric("token_force_loss_mean", original_loss.mean().item())
            self.add_wandb_metric("token_force_loss_std", original_loss.std().item())
            self.add_wandb_metric("token_force_loss_min", original_loss.min().item())

        # Get current refusal beta value based on schedule
        current_beta = self.get_current_refusal_beta()
        
        # If using wandb, log the current beta value
        if self.using_wandb:
            self.add_wandb_metric("refusal_beta", current_beta)

        # Compute refusal direction term if enabled and activations are available
        refusal_term = None
        if self.config.use_refusal_direction and hasattr(self, 'activations') and 'target_layer' in self.activations:
            activations = self.activations['target_layer']
            seq_len = activations.shape[1]
            
            # Debug logging
            if self.using_wandb:
                self.add_wandb_metric("activation_seq_len", seq_len)
            
            if seq_len == 0:
                logger.warning("No activations available (sequence length is 0)")
                return original_loss
            
            if self.config.use_final_prompt_token:
                # When using prefix caching, the activation sequence might not include
                # the 'before' tokens, only: optim_tokens + after_tokens + target_tokens
                # So we need to calculate the position differently
                
                target_length = self.target_ids.shape[1]
                
                # The prompt ends just before the target begins
                # In the activation sequence (which might not include prefix cached tokens),
                # this is at position (seq_len - target_length - 1)
                prompt_end_position = seq_len - target_length - 1
                
                # Ensure the position is valid
                if prompt_end_position < 0 or prompt_end_position >= seq_len:
                    # If we can't find the right position, use the last token before target
                    # or the first available token as fallback
                    logger.warning(f"Calculated prompt end position {prompt_end_position} is out of bounds [0, {seq_len}). "
                                f"Activation seq_len: {seq_len}, target_length: {target_length}")
                    if seq_len > target_length:
                        # Use the token just before where targets would start
                        prompt_end_position = seq_len - target_length - 1
                        prompt_end_position = max(0, prompt_end_position)
                    else:
                        # Sequence is too short, use first token
                        prompt_end_position = 0
                
                # Extract the activation at the prompt end position
                token_activations = activations[:, prompt_end_position:(prompt_end_position + 1), :]
                
                if self.using_wandb:
                    self.add_wandb_metric("prompt_position_used", prompt_end_position)
                    self.add_wandb_metric("target_length", target_length)
                    self.add_wandb_metric("final_prompt_token_shape", list(token_activations.shape))
            else:
                # Original behavior: use first N tokens of the response
                # But make sure we don't exceed the sequence length
                num_tokens = min(self.config.refusal_num_tokens, seq_len)
                token_activations = activations[:, :num_tokens, :]
                
                if self.using_wandb:
                    self.add_wandb_metric("num_response_tokens_used", num_tokens)
            
            # Check if we have valid activations
            if token_activations.shape[1] == 0:
                logger.warning("No valid token activations extracted")
                return original_loss
            
            # Average across tokens for each example
            mean_activations = torch.mean(token_activations, dim=1)
            
            # Check for NaN in activations
            if torch.isnan(mean_activations).any():
                logger.warning("NaN detected in mean activations")
                return original_loss
            
            refusal_term = mean_activations @ self.refusal_vector 
            
            # Take absolute value for minimization
            refusal_term = torch.abs(refusal_term)
            
            # Check for NaN in refusal term
            if torch.isnan(refusal_term).any():
                logger.warning("NaN detected in refusal term")
                return original_loss

            # Add refusal term metrics for wandb
            if self.using_wandb:
                self.add_wandb_metric("refusal_term_mean", refusal_term.mean().item())
                self.add_wandb_metric("refusal_term_min", refusal_term.min().item())
                self.add_wandb_metric("refusal_term_max", refusal_term.max().item())
                
                # Log which tokens were used
                if self.config.use_final_prompt_token:
                    self.add_wandb_metric("refusal_token_type", "final_prompt_token")
                else:
                    self.add_wandb_metric("refusal_token_type", f"first_{num_tokens}_response_tokens")
        
        # Combine losses or return just the original loss
        if self.config.use_refusal_direction and refusal_term is not None:
            combined_loss = (1 - current_beta) * original_loss + current_beta * refusal_term
            
            # Log combined loss metrics for wandb
            if self.using_wandb:
                self.add_wandb_metric("combined_loss_mean", combined_loss.mean().item())
                self.add_wandb_metric("combined_loss_min", combined_loss.min().item())
            
            return combined_loss
        else:
            return original_loss


    def get_current_refusal_beta(self):
        """Get the current refusal beta value based on the selected schedule."""
        
        # If not using refusal direction or scheduling, just return the fixed beta
        if not self.config.use_refusal_direction or not self.config.use_refusal_beta_schedule:
            return self.config.refusal_beta
        
        # Determine end step for schedule
        end_step = self.config.schedule_end_step if self.config.schedule_end_step is not None else self.config.num_steps
        
        # Make sure start_step is less than end_step
        start_step = min(self.config.schedule_start_step, end_step - 1)
        
        # If we're before the start of the schedule, use the start value
        if self.step < start_step:
            return self.config.refusal_beta_start
        
        # If we're after the end of the schedule, use the end value
        if self.step >= end_step:
            return self.config.refusal_beta_end
        
        # Determine current step within the schedule
        current_schedule_step = self.step - start_step
        total_schedule_steps = end_step - start_step
        
        # Apply the appropriate schedule
        if self.config.schedule_type.lower() == "linear":
            from nanogcg.utils import linear_schedule
            return linear_schedule(
                self.config.refusal_beta_start,
                self.config.refusal_beta_end,
                current_schedule_step,
                total_schedule_steps
            )
        elif self.config.schedule_type.lower() == "exponential":
            from nanogcg.utils import exponential_schedule
            return exponential_schedule(
                self.config.refusal_beta_start,
                self.config.refusal_beta_end,
                current_schedule_step,
                total_schedule_steps,
                exponent=self.config.schedule_exponent
            )
        elif self.config.schedule_type.lower() == "power":
            from nanogcg.utils import power_schedule
            return power_schedule(
                self.config.refusal_beta_start,
                self.config.refusal_beta_end,
                current_schedule_step,
                total_schedule_steps,
                power=self.config.schedule_power
            )
        else:
            logger.warning(f"Unknown schedule type: {self.config.schedule_type}. Using fixed beta.")
            return self.config.refusal_beta
        

# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    """Generates a single optimized string using GCG.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.

    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg = GCG(model, tokenizer, config)
    result = gcg.run(messages, target)
    return result
