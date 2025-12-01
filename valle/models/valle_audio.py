# Copyright    2025                             (authors: Sunan Zou)
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
Audio Language Model - Audio-to-Audio generation with AR + NAR structure
Similar to VALLE but takes audio tokens as conditioning instead of text/MIDI.

- Input: Audio tokens (prompt) instead of text/MIDI tokens
- Output: Audio tokens (continuation)
- AR (1st quantizer) + NAR (remaining quantizers) paradigm
"""

import random
from typing import Dict, Iterator, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from icefall.utils import make_pad_mask
from torchmetrics.classification import MulticlassAccuracy

from valle.modules.embedding import SinePositionalEmbedding, TokenEmbedding
from valle.modules.transformer import (
    AdaptiveLayerNorm,
    LayerNorm,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from .macros import NUM_AUDIO_TOKENS

class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)


class VALLE_Audio(nn.Module):
    """
    Audio Language Model - Audio continuation with AR + NAR structure.
    
    Architecture:
    1. AR Decoder: Takes prompt audio, predicts 1st quantizer autoregressively
    2. NAR Decoder: Takes 1st quantizer, predicts remaining quantizers in parallel
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        norm_first: Whether to use pre-normalization
        add_prenet: Whether to add PreNet
        num_quantizers: Number of audio quantizers (default 4 for 32kHz EnCodec)
        nar_scale_factor: Scale factor for NAR decoder dimensions
        prepend_bos: Whether to prepend BOS token for AR
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        num_quantizers: int = 4,
        nar_scale_factor: float = 1.0,
        prepend_bos: bool = False,
    ):
        super().__init__()
        
        nar_d_model = int(d_model * nar_scale_factor)
        
        # AR: Target audio embedding (first quantizer only)
        # +1 for PAD, +1 for BOS
        self.ar_audio_prepend_bos = prepend_bos
        self.ar_audio_embedding = TokenEmbedding(
            d_model, NUM_AUDIO_TOKENS + 1 + int(prepend_bos)
        )       
        self.ar_prompt_embeddings = nn.ModuleList([
            TokenEmbedding(d_model, NUM_AUDIO_TOKENS)
            for _ in range(num_quantizers)
        ])
        
        # PreNet
        if add_prenet:
            self.ar_text_prenet = nn.Sequential(
                Transpose(),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                Transpose(),
                nn.Linear(d_model, d_model),
            )

            self.ar_audio_prenet = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, d_model),
            )
        else:
            self.ar_prompt_prenet = nn.Identity()
            self.ar_audio_prenet = nn.Identity()

        # AR Positional encodings
        self.ar_prompt_position = SinePositionalEmbedding(
            d_model,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.ar_audio_position = SinePositionalEmbedding(
            d_model,
            dropout=0.1,
            scale=False,
            alpha=True,
        )

        # AR Decoder (Transformer Encoder with causal mask)
        self.ar_decoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=LayerNorm(d_model) if norm_first else None,
        )
        self.ar_predict_layer = nn.Linear(
            d_model, NUM_AUDIO_TOKENS + 1, bias=False
        )

        self.ar_accuracy_metric = MulticlassAccuracy(
            NUM_AUDIO_TOKENS + 1,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=NUM_AUDIO_TOKENS,
        )        
        self.rng = random.Random(0)
        self.num_heads = nhead
        self.num_quantizers = num_quantizers
        
        
        # NAR components (for remaining quantizers)
        assert num_quantizers >= 1
        if num_quantizers > 1:
            # NAR: Prompt audio embeddings
            self.nar_prompt_embeddings = nn.ModuleList([
                TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS)
                for _ in range(num_quantizers)
            ])
            
            # NAR: Target audio embeddings
            self.nar_audio_embeddings = nn.ModuleList(
                [TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS + 1)]
                + [
                    TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS)
                    for i in range(num_quantizers - 1)
                ]
            )

            # PreNet
            if add_prenet:
                self.nar_text_prenet = nn.Sequential(
                    Transpose(),
                    nn.Conv1d(
                        nar_d_model, nar_d_model, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Conv1d(
                        nar_d_model, nar_d_model, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Conv1d(
                        nar_d_model, nar_d_model, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    Transpose(),
                    nn.Linear(nar_d_model, nar_d_model),
                )
                self.nar_audio_prenet = nn.Sequential(
                    nn.Linear(nar_d_model, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, nar_d_model),
                )
            else:
                self.nar_prompt_prenet = nn.Identity()
                self.nar_audio_prenet = nn.Identity()
            
            # NAR Positional encodings
            self.nar_prompt_position = SinePositionalEmbedding(
                nar_d_model,
                dropout=0.1,
                scale=False,
                alpha=False,
            )
            self.nar_audio_position = SinePositionalEmbedding(
                nar_d_model,
                dropout=0.1,
                scale=False,
                alpha=False,
            )
            
            # NAR Decoder
            self.nar_decoder = TransformerEncoder(
                TransformerEncoderLayer(
                    nar_d_model,
                    int(nhead * nar_scale_factor),
                    dim_feedforward=nar_d_model * 4,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=norm_first,
                    adaptive_layer_norm=True,
                ),
                num_layers=int(num_layers * nar_scale_factor),
                norm=AdaptiveLayerNorm(
                    nar_d_model, norm=nn.LayerNorm(nar_d_model)
                )
                if norm_first
                else None,
            )
            
            # NAR prediction layers
            self.nar_predict_layers = nn.ModuleList(
                [
                    nn.Linear(nar_d_model, NUM_AUDIO_TOKENS, bias=False)
                    for i in range(num_quantizers - 1)
                ]
            )
            
            # NAR stage embeddings
            self.nar_stage_embeddings = nn.ModuleList(
                [
                    TokenEmbedding(nar_d_model, 1)
                    for i in range(num_quantizers - 1)
                ]
            )
            
            # NAR accuracy metric
            self.nar_accuracy_metric = MulticlassAccuracy(
                NUM_AUDIO_TOKENS + 1,
                top_k=10,
                average="micro",
                multidim_average="global",
                ignore_index=NUM_AUDIO_TOKENS,
            )

    def stage_parameters(self, stage: int = 1) -> Iterator[nn.Parameter]:
        """Get parameters for specific training stage."""
        assert stage > 0
        if stage == 1:
            for name, param in self.named_parameters():
                if name.startswith("ar_"):
                    yield param
        if stage == 2:
            for name, param in self.named_parameters():
                if name.startswith("nar_"):
                    yield param

    def stage_named_parameters(self, stage: int = 1):
        """Get named parameters for specific training stage."""
        assert stage > 0
        if stage == 1:
            for pair in self.named_parameters():
                if pair[0].startswith("ar_"):
                    yield pair
        if stage == 2:
            for pair in self.named_parameters():
                if pair[0].startswith("nar_"):
                    yield pair

    def pad_y_eos(self, y, y_mask_int, eos_id):
        """Add EOS token and prepare input/target pairs."""
        targets = F.pad(y, (0, 1), value=0) + eos_id * F.pad(
            y_mask_int, (0, 1), value=1
        )
        # inputs, targets
        if self.ar_audio_prepend_bos:
            return (
                F.pad(targets[:, :-1], (1, 0), value=NUM_AUDIO_TOKENS + 1),
                targets,
            )
        return targets[:, :-1], targets[:, 1:]

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: torch.Tensor,
        reduction: str = "sum",
        train_stage: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass for training.
        
        Args:
            x: Prompt audio tokens (B, S, Q) - conditioning input
            x_lens: Prompt lengths (B,)
            y: Target audio tokens (B, T, Q) - what we want to generate
            y_lens: Target lengths (B,)
            reduction: Loss reduction method
            train_stage: 0=both, 1=AR only, 2=NAR only
            
        Returns:
            predictions, loss, metrics
        """
        assert x.ndim == 3, x.shape  # (B, S, Q)
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape  # (B, T, Q)
        assert y_lens.ndim == 1, y_lens.shape
        
        # Create masks
        x_mask = make_pad_mask(x_lens).to(x.device)
        y_mask = make_pad_mask(y_lens).to(y.device)
        y_mask_int = y_mask.type(torch.int64)
        

        # Convert to int64 for embeddings
        x = x.type(torch.int64)
        prompt = self.ar_prompt_embeddings[0](x[..., 0])
        for q in range(1, self.num_quantizers):
            prompt = prompt + self.ar_prompt_embeddings[q](x[..., q])
        
        prompt = self.ar_audio_prenet(prompt)
        prompt = self.ar_audio_position(prompt)
        

        
        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))
        
        # ========== AR Stage: Predict 1st quantizer ==========
        y_ar, targets = self.pad_y_eos(
            codes[..., 0], y_mask_int, eos_id=NUM_AUDIO_TOKENS
        )
        
        # Concatenate prompt and target
        x_len = x_lens.max()

        metrics = {}
        total_loss = 0.0

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
        if self.ar_audio_prepend_bos:
            ar_xy_padding_mask = torch.concat(
                [x_mask, F.pad(y_mask, (1, 0), value=False)], dim=1
            )
        else:
            ar_xy_padding_mask = xy_padding_mask
        if train_stage in [0, 1]:
            y_len = y_lens.max() + int(self.ar_audio_prepend_bos)

            x_attn_mask = F.pad(
                torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
                (0, y_len),
                value=True,  # Prompt cannot see target
            )
            y_attn_mask = F.pad(
                torch.triu(
                    torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                    diagonal=1,
                ),
                (x_len, 0),
                value=False,  # Target can see all prompt
            )
            xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
            
            # Expand for multi-head attention
            bsz = x.shape[0]
            src_len = x_len + y_len
            _xy_padding_mask = (
                ar_xy_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(bsz * self.num_heads, 1, src_len)
            )

            xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)

            # Convert to float mask
            new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=prompt.dtype)
            new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
            xy_attn_mask = new_attn_mask

            y_emb = self.ar_audio_embedding(y_ar)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)

            xy_pos = torch.concat([prompt, y_pos], dim=1)
            # Forward through AR decoder
            xy_dec, _ = self.ar_decoder(
                (xy_pos, None),
                mask=xy_attn_mask,
            )
            # Get predictions (only for target part)
            logits = self.ar_predict_layer(xy_dec[:, x_len:]).permute(0, 2, 1)
            # loss
            total_loss = F.cross_entropy(logits, targets, reduction=reduction)

            metrics["ArTop10Accuracy"] = self.ar_accuracy_metric(
                logits.detach(), targets
            ).item() * y_lens.sum().type(torch.float32)

        if self.num_quantizers == 1:
            return ((prompt, codes), total_loss, metrics)

        # ========== NAR Stage: Predict remaining quantizers ==========
        if self.ar_audio_prepend_bos:
            y_ar = y_ar[:, 1:]
        
        if train_stage in [0, 2]:
            num_nar_layers = self.num_quantizers - 1
            nar_stage = self.rng.choices(
                [_k for _k in range(1, self.num_quantizers)],
                weights=[1.0 / num_nar_layers] * num_nar_layers,
                k=1,
            )[0]

            # Embed NAR prompt audio
            nar_prompt = self.nar_prompt_embeddings[0](x[..., 0])
            for q in range(1, self.num_quantizers):
                nar_prompt = nar_prompt + self.nar_prompt_embeddings[q](x[..., q])
            
            nar_prompt = self.nar_prompt_prenet(nar_prompt)
            nar_prompt = self.nar_prompt_position(nar_prompt)
            # Embed target audio (sum up to current stage)
            y_emb = self.nar_audio_embeddings[0](y_ar)
            for j in range(1, nar_stage):
                y_emb = y_emb + self.nar_audio_embeddings[j](codes[..., j])
            y_len = y_lens.max()
            targets_nar = codes[..., nar_stage] + NUM_AUDIO_TOKENS * y_mask_int
            xy_padding_mask = torch.concat(
                [
                    x_mask,
                    F.pad(y_mask, (y_emb.shape[1] - y_len, 0), value=False),

                ],
                dim=1,
            )
            y_pos = self.nar_audio_prenet(y_emb)
            y_pos = self.nar_audio_position(y_pos)
            # Concatenate prompt and target
            xy_pos = torch.concat([nar_prompt, y_pos], dim=1)
        
            # Forward through NAR decoder
            xy_dec, _ = self.nar_decoder(
                (xy_pos, self.nar_stage_embeddings[nar_stage - 1].weight),
                src_key_padding_mask=xy_padding_mask,
            )
            
            # Get predictions (only for target part)
            xy_dec = xy_dec[:, x_lens.max():]
            logits = self.nar_predict_layers[nar_stage - 1](xy_dec).permute(0, 2, 1)
            
            # Compute NAR loss
            total_length = (y_lens).sum().type(torch.float32)
            total_loss += (
                F.cross_entropy(
                    logits,
                    targets_nar,
                    ignore_index=NUM_AUDIO_TOKENS,
                    reduction=reduction,
                )
            )
            metrics["NarTop10Accuracy"] = (
                self.nar_accuracy_metric(
                    F.pad(
                        logits.detach(),
                        (0, 0, 0, 1, 0, 0),
                        value=logits.min().cpu().item(),
                    ),
                    targets_nar,
                ).item()
                * total_length
            )

        if train_stage == 0:
            total_loss = total_loss / 2.0

        return ((prompt, codes), total_loss, metrics)

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        max_new_tokens: int = 500,
        top_k: int = -100,
        temperature: float = 1.0,
        return_full_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Generate audio tokens autoregressively.
        
        Args:
            x: Prompt audio tokens (1, S, Q)
            x_lens: Prompt lengths (1,)
            max_new_tokens: Maximum number of tokens to generate
            top_k: Top-k sampling
            temperature: Sampling temperature
            
        Returns:
            Generated audio tokens (1, S+T, Q)
        """
        from valle.models.valle import topk_sampling
        
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert x.shape[0] == 1, "Only batch_size=1 supported"
        
        device = x.device
        
        x_original = x.clone()

        # Convert to int64 for embeddings
        x = x.type(torch.int64)
        
        # Embed prompt audio (sum over all quantizers)
        prompt = self.ar_prompt_embeddings[0](x[..., 0])
        for q in range(1, self.num_quantizers):
            prompt = prompt + self.ar_prompt_embeddings[q](x[..., q])
        
        prompt = self.ar_audio_prenet(prompt)
        prompt = self.ar_audio_position(prompt)
        
        x_len = x_lens.max()
        x_mask = make_pad_mask(x_lens).to(device)
        
        # ========== AR Generation: 1st quantizer ==========
        y = x[..., 0][:, :0]  # Empty tensor to start
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=NUM_AUDIO_TOKENS + 1)
        
        prefix_len = x.shape[1]
        
        while True:
            y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)
            
            xy_pos = torch.concat([prompt, y_pos], dim=1)
            
            y_len = y.shape[1]
            x_attn_mask = F.pad(
                torch.zeros((x_len, x_len), dtype=torch.bool, device=device),
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(
                    torch.ones(y_len, y_len, dtype=torch.bool, device=device),
                    diagonal=1,
                ),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
            
            xy_dec, _ = self.ar_decoder(
                (xy_pos, None),
                mask=xy_attn_mask,
            )
            
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(
                logits, top_k=top_k, top_p=1.0, temperature=temperature
            )
            
            # Check for EOS or max length
            if (
                torch.argmax(logits, dim=-1)[0] == NUM_AUDIO_TOKENS
                or samples[0, 0] == NUM_AUDIO_TOKENS
                or y.shape[1] > max_new_tokens
            ):
                print(f"AR stage: generated {y.shape[1]} tokens")
                break
            
            y = torch.concat([y, samples], dim=1)
        
        codes = [y[:, int(self.ar_audio_prepend_bos):]]
        
        if self.num_quantizers == 1:
            return torch.stack(codes, dim=-1)
        
        # ========== NAR Generation: Remaining quantizers ==========
        y_ar = y[:, int(self.ar_audio_prepend_bos):]
        
        # Embed NAR prompt
        nar_prompt = self.nar_prompt_embeddings[0](x[..., 0])
        for q in range(1, self.num_quantizers):
            nar_prompt = nar_prompt + self.nar_prompt_embeddings[q](x[..., q])
        
        nar_prompt = self.nar_prompt_prenet(nar_prompt)
        nar_prompt = self.nar_prompt_position(nar_prompt)
        
        # Start with first quantizer
        y_emb = self.nar_audio_embeddings[0](y_ar)
        
        for i, (predict_layer, embedding_layer) in enumerate(
            zip(self.nar_predict_layers, self.nar_audio_embeddings[1:])
        ):
            y_pos = self.nar_audio_prenet(y_emb)
            y_pos = self.nar_audio_position(y_pos)
            xy_pos = torch.concat([nar_prompt, y_pos], dim=1)
            
            xy_dec, _ = self.nar_decoder(
                (xy_pos, self.nar_stage_embeddings[i].weight)
            )
            
            logits = predict_layer(xy_dec[:, prefix_len:])
            samples = torch.argmax(logits, dim=-1)
            codes.append(samples)
            
            # Add to embeddings for next stage
            if i < self.num_quantizers - 2:
                y_emb = y_emb + embedding_layer(samples)
        
        assert len(codes) == self.num_quantizers
        
        # Stack codes: (1, T, Q)
        # Each element in codes has shape (1, T), stacking along last dim gives (1, T, Q)
        generated_codes = torch.stack(codes, dim=-1)
        
        # Ensure generated_codes matches the dtype of x_original
        if generated_codes.dtype != x_original.dtype:
            generated_codes = generated_codes.to(dtype=x_original.dtype)
        
        # Optionally concatenate with prompt
        if return_full_sequence:
            # x_original shape: (1, S, Q), generated_codes shape: (1, T, Q)
            # Concatenate along sequence dimension to get (1, S+T, Q)
            return torch.cat([x_original, generated_codes], dim=1)
        else:
            return generated_codes
