import torch
import logging
from typing import Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger("nanogcg")

class IRISRefusalHandler:
    """Handles IRIS-style multi-layer refusal direction computations."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.activations = {}
        self.hooks = []
        
        # Load refusal vectors
        self.refusal_vectors = self._load_refusal_vectors()
        
        # Register hooks if using refusal direction
        if config.use_refusal_direction:
            self.register_activation_hooks()
    
    def _load_refusal_vectors(self) -> Dict[int, torch.Tensor]:
        """Load refusal vectors for specified layers."""
        refusal_vectors = {}
        
        if self.config.refusal_vector is not None:
            # If a single vector is provided, we'll use it for specified layers
            base_vector = self.config.refusal_vector.to(self.model.device, self.model.dtype)
        elif self.config.refusal_vector_path:
            # Load from file
            try:
                data = torch.load(self.config.refusal_vector_path, map_location=self.model.device)
                if isinstance(data, dict):
                    # Assume dict with layer indices as keys
                    base_vector = None
                    refusal_vectors = {int(k): v.to(self.model.device, self.model.dtype) for k, v in data.items()}
                else:
                    # Single vector
                    base_vector = data.to(self.model.device, self.model.dtype)
            except:
                # Try numpy
                base_vector = torch.tensor(np.load(self.config.refusal_vector_path), 
                                         device=self.model.device, dtype=self.model.dtype)
        else:
            raise ValueError("Must provide either refusal_vector or refusal_vector_path")
        
        # If we have a base vector, create layer-specific vectors
        if base_vector is not None and not refusal_vectors:
            # Use layers specified in config or default to ALL layers
            if hasattr(self.config, 'refusal_layers') and self.config.refusal_layers is not None:
                layers = self.config.refusal_layers
            else:
                # Use all available layers
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    num_layers = len(self.model.model.layers)
                    layers = list(range(num_layers))
                    logger.info(f"No specific layers configured - using all {num_layers} model layers")
            
            for layer_idx in layers:
                # Normalize the vector
                normalized_vector = base_vector / (torch.norm(base_vector) + 1e-8)
                refusal_vectors[layer_idx] = normalized_vector
        
        # Normalize all vectors
        for layer_idx in refusal_vectors:
            refusal_vectors[layer_idx] = refusal_vectors[layer_idx] / (torch.norm(refusal_vectors[layer_idx]) + 1e-8)
            
        logger.info(f"Loaded refusal vectors for layers: {list(refusal_vectors.keys())}")
        return refusal_vectors
    
    def register_activation_hooks(self):
        """Register hooks to extract activations from specified layers."""
        
        def get_activation(name, layer_idx):
            def hook(module, input, output):
                # Store the residual stream activation (input to the layer)
                if isinstance(input, tuple):
                    activation = input[0]
                else:
                    activation = input
                self.activations[f'layer_{layer_idx}'] = activation
            return hook
        
        # Clear any existing hooks
        self.remove_hooks()
        
        # Register hooks on layers that have refusal vectors
        for layer_idx in self.refusal_vectors.keys():
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                if layer_idx < len(self.model.model.layers):
                    # Hook the input_layernorm to get residual stream
                    layer = self.model.model.layers[layer_idx]
                    if hasattr(layer, 'input_layernorm'):
                        hook = layer.input_layernorm.register_forward_hook(
                            get_activation(f'layer_{layer_idx}', layer_idx)
                        )
                        self.hooks.append(hook)
                    else:
                        logger.warning(f"Layer {layer_idx} doesn't have input_layernorm")
        
        logger.info(f"Registered {len(self.hooks)} activation hooks")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_iris_refusal_loss(self, last_input_token_position: int) -> Optional[torch.Tensor]:
        """
        Compute IRIS-style refusal loss: β ∑(r̂ᵀa)²
        
        Args:
            last_input_token_position: Position of the last input token in the sequence
            
        Returns:
            Refusal loss term or None if no activations available
        """
        if not self.activations:
            return None
        
        total_squared_alignment = 0.0
        num_terms = 0
        
        # Iterate through each layer that has both activations and refusal vector
        for layer_idx, refusal_vector in self.refusal_vectors.items():
            activation_key = f'layer_{layer_idx}'
            
            if activation_key not in self.activations:
                continue
                
            activation = self.activations[activation_key]
            
            # Extract activation at the last input token position
            if last_input_token_position >= activation.shape[1]:
                logger.warning(f"Token position {last_input_token_position} out of bounds for layer {layer_idx}")
                continue
                
            # Get the activation at the specified token position for all samples in batch
            token_activation = activation[:, last_input_token_position, :]  # [batch_size, hidden_dim]
            
            # Ensure dimensions match
            if token_activation.shape[-1] != refusal_vector.shape[0]:
                logger.warning(f"Dimension mismatch at layer {layer_idx}: "
                             f"activation {token_activation.shape[-1]} vs refusal {refusal_vector.shape[0]}")
                continue
            
            # Compute dot product for each sample in batch
            dot_products = torch.matmul(token_activation, refusal_vector)  # [batch_size]
            
            # Square the dot products and add to total
            squared_dots = dot_products ** 2
            total_squared_alignment = total_squared_alignment + squared_dots
            num_terms += 1
        
        if num_terms == 0:
            logger.warning("No valid layer activations found for refusal loss computation")
            return None
            
        # Average across layers (optional - IRIS sums across all terms)
        if self.config.average_across_layers:
            total_squared_alignment = total_squared_alignment / num_terms
            
        return total_squared_alignment / self.config.iris_normalisation_constant
    
    def get_last_input_token_position(self, seq_len: int, target_len: int) -> int:
        """
        Determine the position of the last input token.
        
        Args:
            seq_len: Total sequence length of activations
            target_len: Length of target tokens
            
        Returns:
            Position of the last input token
        """
        # The last input token is the one right before the target starts
        # In the activation sequence: optim_tokens + after_tokens + target_tokens
        # Last input position = seq_len - target_len - 1
        
        last_pos = seq_len - target_len - 1
        
        # Ensure valid position
        if last_pos < 0:
            logger.warning(f"Calculated negative last position {last_pos}, using 0")
            return 0
        elif last_pos >= seq_len:
            logger.warning(f"Calculated position {last_pos} >= seq_len {seq_len}, using {seq_len-1}")
            return seq_len - 1
            
        return last_pos
