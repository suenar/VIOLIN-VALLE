# Copyright    2024                             (authors: Your Name)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Audio Language Model - Audio-only autoregressive generation
Takes audio tokens as input and generates continuations autoregressively.
"""

import random
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from icefall.utils import make_pad_mask
from torchmetrics.classification import MulticlassAccuracy

from valle.modules.embedding import SinePositionalEmbedding, TokenEmbedding
from valle.modules.transformer import (
    LayerNorm,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from .macros import NUM_AUDIO_TOKENS


class AudioLM(nn.Module):
    """
    Audio Language Model for autoregressive audio generation.
    
    Unlike VALLE which uses text/MIDI conditioning, this model:
    - Takes only audio tokens as input (3-10 seconds)
    - Generates audio tokens autoregressively like a language model
    - Uses causal masking for autoregressive generation
    - Can be used for audio continuation, in-filling, or unconditional generation
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        norm_first: Whether to use pre-normalization
        add_prenet: Whether to add PreNet
        num_quantizers: Number of audio quantizers (default 4 for 32kHz EnCodec)
        max_seq_len: Maximum sequence length for positional encoding
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        num_quantizers: int = 4,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_quantizers = num_quantizers
        self.max_seq_len = max_seq_len
        
        # Audio token embeddings for each quantizer level
        # +1 for padding token, +1 for BOS token
        self.audio_embeddings = nn.ModuleList([
            TokenEmbedding(d_model, NUM_AUDIO_TOKENS + 2)  # +2 for PAD and BOS
            for _ in range(num_quantizers)
        ])
        
        # PreNet (optional)
        if add_prenet:
            self.prenet = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(0.25),
            )
        else:
            self.prenet = nn.Identity()
        
        # Positional encoding
        self.position_encoding = SinePositionalEmbedding(
            d_model,
            dropout=dropout,
            scale=False,
            alpha=True,
        )
        
        # Transformer encoder (used as decoder with causal masking)
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=LayerNorm(d_model) if norm_first else None,
        )
        
        # Output projection layers for each quantizer
        self.output_layers = nn.ModuleList([
            nn.Linear(d_model, NUM_AUDIO_TOKENS + 1, bias=False)  # +1 for EOS
            for _ in range(num_quantizers)
        ])
        
        # Metrics
        self.accuracy_metrics = nn.ModuleList([
            MulticlassAccuracy(
                NUM_AUDIO_TOKENS + 1,
                top_k=10,
                average="micro",
                multidim_average="global",
                ignore_index=NUM_AUDIO_TOKENS,
            )
            for _ in range(num_quantizers)
        ])
        
        self.rng = random.Random(0)
        self.num_heads = nhead

    def _prepare_input_embeddings(
        self, 
        audio_tokens: torch.Tensor,
        token_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepare input embeddings by summing embeddings from all quantizer levels.
        
        Args:
            audio_tokens: (B, T, Q) where Q is num_quantizers
            token_mask: (B, T) binary mask
            
        Returns:
            embeddings: (B, T, D)
        """
        # Sum embeddings from all quantizer levels
        embeddings = self.audio_embeddings[0](audio_tokens[..., 0])
        for q in range(1, self.num_quantizers):
            embeddings = embeddings + self.audio_embeddings[q](audio_tokens[..., q])
        
        # Apply prenet
        embeddings = self.prenet(embeddings)
        
        # Add positional encoding
        embeddings = self.position_encoding(embeddings)
        
        return embeddings

    def forward(
        self,
        audio_tokens: torch.Tensor,
        audio_lens: torch.Tensor,
        reduction: str = "sum",
        target_quantizer: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass for training.
        
        Args:
            audio_tokens: (B, T, Q) audio tokens for all quantizers
            audio_lens: (B,) lengths of each sequence before padding
            reduction: Loss reduction method ("sum", "mean", "none")
            target_quantizer: Which quantizer to predict (None = all sequentially)
            
        Returns:
            logits: Predicted logits
            loss: Cross-entropy loss
            metrics: Dictionary of metrics
        """
        batch_size, seq_len, num_q = audio_tokens.shape
        assert num_q == self.num_quantizers
        
        # Create padding mask
        padding_mask = make_pad_mask(audio_lens).to(audio_tokens.device)
        
        # Prepare targets: shift by 1 for autoregressive prediction
        # Input: [BOS, tok1, tok2, ..., tokN]
        # Target: [tok1, tok2, ..., tokN, EOS]
        targets = audio_tokens.clone()
        
        # Add BOS token at the beginning (use NUM_AUDIO_TOKENS + 1 as BOS)
        bos_token = torch.full(
            (batch_size, 1, num_q), 
            NUM_AUDIO_TOKENS + 1, 
            dtype=audio_tokens.dtype,
            device=audio_tokens.device
        )
        inputs = torch.cat([bos_token, audio_tokens[:, :-1, :]], dim=1)
        
        # Pad targets with EOS token (NUM_AUDIO_TOKENS)
        targets = F.pad(targets, (0, 0, 0, 1), value=NUM_AUDIO_TOKENS)
        targets = targets[:, 1:, :]  # Shift and take [1:] to match input length
        
        # Update padding mask for the added BOS token
        input_padding_mask = F.pad(padding_mask, (1, 0), value=False)
        
        # Get input embeddings
        embeddings = self._prepare_input_embeddings(inputs, input_padding_mask)
        
        # Create causal attention mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=audio_tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        
        # Expand causal mask for multi-head attention
        causal_mask = causal_mask.unsqueeze(0).expand(
            batch_size * self.num_heads, -1, -1
        )
        
        # Add padding mask to causal mask
        causal_mask = causal_mask.clone()
        padding_mask_expanded = input_padding_mask.unsqueeze(1).expand(-1, seq_len, -1)
        padding_mask_expanded = padding_mask_expanded.repeat_interleave(self.num_heads, dim=0)
        causal_mask = causal_mask.logical_or(padding_mask_expanded)
        
        # Convert boolean mask to float mask for transformer
        float_mask = torch.zeros_like(causal_mask, dtype=embeddings.dtype)
        float_mask.masked_fill_(causal_mask, float("-inf"))
        
        # Forward through transformer
        hidden_states, _ = self.transformer(
            (embeddings, None),
            mask=float_mask,
        )
        
        # Compute loss for each quantizer
        total_loss = 0.0
        metrics = {}
        
        # Determine which quantizers to train
        if target_quantizer is not None:
            quantizers_to_train = [target_quantizer]
        else:
            # Train all quantizers
            quantizers_to_train = list(range(self.num_quantizers))
        
        for q in quantizers_to_train:
            # Get predictions for this quantizer
            logits = self.output_layers[q](hidden_states).permute(0, 2, 1)  # (B, C, T)
            targets_q = targets[..., q]  # (B, T)
            
            # Compute loss (masked by padding)
            loss = F.cross_entropy(
                logits,
                targets_q,
                ignore_index=NUM_AUDIO_TOKENS,
                reduction=reduction,
            )
            
            total_loss += loss
            
            # Compute accuracy
            accuracy = self.accuracy_metrics[q](logits.detach(), targets_q)
            metrics[f"Quantizer{q}Accuracy"] = accuracy.item() * audio_lens.sum().type(torch.float32)
        
        # Average loss across quantizers
        if len(quantizers_to_train) > 1:
            total_loss = total_loss / len(quantizers_to_train)
        
        return hidden_states, total_loss, metrics

    def inference(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = -100,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate audio tokens autoregressively.
        
        Args:
            prompt_tokens: (1, T, Q) prompt audio tokens
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            generated_tokens: (1, T+max_new_tokens, Q) generated audio tokens
        """
        self.eval()
        device = prompt_tokens.device
        batch_size = prompt_tokens.shape[0]
        assert batch_size == 1, "Only batch_size=1 supported for inference"
        
        # Start with prompt
        generated = prompt_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get current sequence length
                cur_len = generated.shape[1]
                
                # Prepare input with BOS token
                bos_token = torch.full(
                    (1, 1, self.num_quantizers),
                    NUM_AUDIO_TOKENS + 1,
                    dtype=generated.dtype,
                    device=device
                )
                inputs = torch.cat([bos_token, generated], dim=1)
                
                # Get embeddings
                embeddings = self._prepare_input_embeddings(
                    inputs, 
                    torch.zeros(1, cur_len + 1, dtype=torch.bool, device=device)
                )
                
                # Create causal mask
                seq_len = cur_len + 1
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                    diagonal=1,
                )
                causal_mask = causal_mask.unsqueeze(0).expand(self.num_heads, -1, -1)
                float_mask = torch.zeros_like(causal_mask, dtype=embeddings.dtype)
                float_mask.masked_fill_(causal_mask, float("-inf"))
                
                # Forward through transformer
                hidden_states, _ = self.transformer(
                    (embeddings, None),
                    mask=float_mask,
                )
                
                # Get predictions for the last token
                last_hidden = hidden_states[:, -1:, :]  # (1, 1, D)
                
                # Generate token for each quantizer
                new_tokens = []
                for q in range(self.num_quantizers):
                    logits = self.output_layers[q](last_hidden).squeeze(1)  # (1, C)
                    
                    # Apply temperature
                    if temperature != 1.0:
                        logits = logits / temperature
                    
                    # Sample next token
                    next_token = self._sample_token(logits, top_k, top_p)
                    new_tokens.append(next_token)
                    
                    # Check for EOS
                    if next_token.item() == NUM_AUDIO_TOKENS:
                        print(f"Generated {cur_len} tokens, stopping at EOS")
                        return generated
                
                # Stack new tokens and append
                new_tokens = torch.stack(new_tokens, dim=-1)  # (1, Q)
                generated = torch.cat([generated, new_tokens.unsqueeze(1)], dim=1)
        
        print(f"Generated {max_new_tokens} tokens, reached max length")
        return generated

    def _sample_token(self, logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        """Sample a token from logits using top-k and top-p sampling."""
        from valle.models.valle import topk_sampling
        return topk_sampling(logits, top_k=top_k, top_p=top_p, temperature=1.0)
