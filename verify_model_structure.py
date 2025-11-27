#!/usr/bin/env python3
"""
Quick script to verify the VALLE_Audio model structure is correct.
This checks that ar_audio_embedding is a TokenEmbedding, not a ModuleList.
"""

import sys
sys.path.insert(0, '/workspace')

import torch
import torch.nn as nn
from valle.models.valle_audio import VALLE_Audio
from valle.modules.embedding import TokenEmbedding

def verify_model_structure():
    """Verify the model structure is correct."""
    print("Creating VALLE_Audio model...")
    model = VALLE_Audio(
        d_model=1024,
        nhead=16,
        num_layers=12,
        norm_first=True,
        add_prenet=False,
        num_quantizers=4,
        nar_scale_factor=1.0,
        prepend_bos=False,
    )
    
    print("\nChecking model structure...")
    
    # Check ar_prompt_embeddings
    if isinstance(model.ar_prompt_embeddings, nn.ModuleList):
        print("✓ ar_prompt_embeddings is ModuleList - CORRECT")
    else:
        print(f"✗ ar_prompt_embeddings is {type(model.ar_prompt_embeddings)} - WRONG")
        return False
    
    if len(model.ar_prompt_embeddings) == 4:
        print(f"✓ ar_prompt_embeddings has 4 embeddings - CORRECT")
    else:
        print(f"✗ ar_prompt_embeddings has {len(model.ar_prompt_embeddings)} embeddings - WRONG")
        return False
    
    # Check ar_audio_embedding
    if isinstance(model.ar_audio_embedding, TokenEmbedding):
        print("✓ ar_audio_embedding is TokenEmbedding - CORRECT")
    elif isinstance(model.ar_audio_embedding, nn.ModuleList):
        print("✗ ar_audio_embedding is ModuleList - WRONG!")
        print("  This should be a single TokenEmbedding, not a ModuleList")
        return False
    else:
        print(f"✗ ar_audio_embedding is {type(model.ar_audio_embedding)} - WRONG")
        return False
    
    # Test forward pass with dummy data
    print("\nTesting forward pass with dummy data...")
    batch_size = 2
    prompt_len = 50
    target_len = 100
    num_quantizers = 4
    
    x = torch.randint(0, 1024, (batch_size, prompt_len, num_quantizers))
    x_lens = torch.tensor([prompt_len, prompt_len])
    y = torch.randint(0, 1024, (batch_size, target_len, num_quantizers))
    y_lens = torch.tensor([target_len, target_len])
    
    try:
        with torch.no_grad():
            predicts, loss, metrics = model(
                x=x,
                x_lens=x_lens,
                y=y,
                y_lens=y_lens,
                reduction="sum",
                train_stage=0,
            )
        print("✓ Forward pass successful!")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Metrics: {metrics}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed with error:")
        print(f"  {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    success = verify_model_structure()
    if success:
        print("\n" + "="*60)
        print("SUCCESS: Model structure is correct!")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("FAILURE: Model structure is incorrect!")
        print("Please check your valle/models/valle_audio.py file")
        print("="*60)
        sys.exit(1)
