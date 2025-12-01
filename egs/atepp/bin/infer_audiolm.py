#!/usr/bin/env python3
# Copyright    2025                            (authors: Sunan Zou)
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
Audio-to-Audio inference script for VALLE-audio model.
This script generates audio continuation given an audio prompt.

Usage example:
    python3 bin/infer_audiolm.py \
        --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
        --audio-prompts ./prompts/prompt_A.wav \
        --output-dir infer/audio_continuation \
        --output-file generated \
        --max-duration 10.0 \
        --top-k -100 \
        --temperature 1.0
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, '/workspace')

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torchaudio
from icefall.utils import AttributeDict, str2bool

from valle.data import (
    AudioTokenizer,
    AudioTokenizer32,
    AudioTokenizer32FT,
    AudioTokenizer32Violin,
    tokenize_audio,
)
from valle.models import get_model


def get_args():
    parser = argparse.ArgumentParser(
        description="Audio-to-audio inference for VALLE-audio model"
    )

    parser.add_argument(
        "--audio-prompts",
        type=str,
        required=True,
        help="Audio prompt file path (single file or multiple files separated by |).",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infer/audio_continuation"),
        help="Path to save generated audio files.",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="generated",
        help="Output file name (without extension).",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=10.0,
        help="Maximum duration (in seconds) to generate.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Top-k sampling for AR decoder (if > 0). Use -100 for unrestricted sampling.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for AR decoder.",
    )

    parser.add_argument(
        "--audio-tokenizer",
        type=str,
        default="Encodec32Violin",
        choices=["Encodec", "Encodec32", "Encodec32FT", "Encodec32Violin"],
        help="Which audio tokenizer to use.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="VALLE-audio",
        help="Model name.",
    )

    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=1024,
        help="Model decoder dimension.",
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=16,
        help="Number of attention heads.",
    )

    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=12,
        help="Number of decoder layers.",
    )

    parser.add_argument(
        "--num-quantizers",
        type=int,
        default=4,
        help="Number of audio quantizers.",
    )

    parser.add_argument(
        "--norm-first",
        type=str2bool,
        default=True,
        help="Whether to use pre-normalization.",
    )

    parser.add_argument(
        "--add-prenet",
        type=str2bool,
        default=False,
        help="Whether to add PreNet.",
    )

    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="NAR scale factor.",
    )

    parser.add_argument(
        "--prepend-bos",
        type=str2bool,
        default=False,
        help="Whether to prepend BOS token.",
    )

    return parser.parse_args()


def load_model(checkpoint_path, args, device):
    """Load VALLE-audio model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model arguments
    model_args = AttributeDict()
    model_args.model_name = args.model_name
    model_args.decoder_dim = args.decoder_dim
    model_args.nhead = args.nhead
    model_args.num_decoder_layers = args.num_decoder_layers
    model_args.norm_first = args.norm_first
    model_args.add_prenet = args.add_prenet
    model_args.scale_factor = args.scale_factor
    model_args.prepend_bos = args.prepend_bos
    model_args.num_quantizers = args.num_quantizers
    
    # Create model
    model = get_model(model_args)
    
    # Load state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logging.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected keys: {unexpected_keys}")
    
    model.to(device)
    model.eval()
    
    logging.info("Model loaded successfully")
    return model


def load_audio_tokenizer(tokenizer_name):
    """Load the appropriate audio tokenizer."""
    if tokenizer_name == "Encodec":
        return AudioTokenizer()
    elif tokenizer_name == "Encodec32":
        return AudioTokenizer32()
    elif tokenizer_name == "Encodec32FT":
        return AudioTokenizer32FT()
    elif tokenizer_name == "Encodec32Violin":
        return AudioTokenizer32Violin()
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")


@torch.no_grad()
def main():
    args = get_args()
    
    # Setup device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"Using device: {device}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, args, device)
    
    # Load audio tokenizer
    audio_tokenizer = load_audio_tokenizer(args.audio_tokenizer)
    logging.info(f"Using audio tokenizer: {args.audio_tokenizer}")
    
    # Process audio prompts
    audio_prompt_files = args.audio_prompts.split("|")
    logging.info(f"Processing {len(audio_prompt_files)} audio prompt(s)")
    
    for idx, audio_file in enumerate(audio_prompt_files):
        if not os.path.exists(audio_file):
            logging.error(f"Audio file not found: {audio_file}")
            continue
        
        logging.info(f"Processing prompt: {audio_file}")
        
        # Tokenize audio prompt
        encoded_frames = tokenize_audio(args.audio_tokenizer, audio_file, audio_tokenizer)
        # encoded_frames shape: (B, T, Q) where B=1
        
        # Move to device
        audio_prompt = encoded_frames.to(device)
        audio_prompt_lens = torch.tensor([audio_prompt.shape[1]], device=device)
        
        logging.info(f"Audio prompt shape: {audio_prompt.shape}")
        logging.info(f"Audio prompt length: {audio_prompt_lens[0]} frames")
        
        # Calculate max tokens to generate
        # EnCodec at 32kHz uses 75 Hz frame rate, so 75 frames = 1 second
        frame_rate = 75  # frames per second for EnCodec at 32kHz
        max_new_tokens = int(args.max_duration * frame_rate)
        logging.info(f"Generating up to {max_new_tokens} new frames (~{args.max_duration}s)")
        
        # Generate audio continuation (full sequence: prompt + generated)
        logging.info("Starting generation...")
        full_sequence = model.inference(
            x=audio_prompt,
            x_lens=audio_prompt_lens,
            max_new_tokens=max_new_tokens,
            top_k=args.top_k,
            temperature=args.temperature
        )
        
        logging.info(f"Full sequence shape: {full_sequence.shape}")
        
        # Decode to audio
        # full_sequence shape: (1, T_total, Q)
        # Need to convert to (1, Q, T_total) for decoder
        full_sequence_transposed = full_sequence.transpose(1, 2)
        
        logging.info("Decoding to audio...")
        audio_samples = audio_tokenizer.decode(full_sequence_transposed, None)
        
        # Save full generated audio (including prompt)
        output_file = args.output_dir / f"{args.output_file}_{idx}_full.wav"
        torchaudio.save(
            str(output_file),
            audio_samples[0].cpu(),
            32000
        )
        logging.info(f"Saved full audio to: {output_file}")
        
        # Also save continuation only (without prompt)
        prompt_len = audio_prompt.shape[1]
        continuation_codes = full_sequence[:, prompt_len:, :]
        continuation_codes_transposed = continuation_codes.transpose(1, 2)
        
        continuation_samples = audio_tokenizer.decode(continuation_codes_transposed, None)
        
        output_file_cont = args.output_dir / f"{args.output_file}_{idx}_continuation.wav"
        torchaudio.save(
            str(output_file_cont),
            continuation_samples[0].cpu(),
            32000
        )
        logging.info(f"Saved continuation to: {output_file_cont}")
        
        logging.info(f"Successfully generated audio for prompt {idx + 1}/{len(audio_prompt_files)}")
    
    logging.info("All done!")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()