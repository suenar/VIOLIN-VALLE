#!/usr/bin/env python3
"""
Visualization script for VALLE model masks.
This script demonstrates the different mask patterns used in VALLE.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def visualize_causal_mask(seq_len=8):
    """Visualize the causal (upper triangular) mask for AR decoder."""
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    # Convert to numpy for visualization
    mask_np = mask.numpy().astype(int)
    
    # Plot
    cmap = ListedColormap(['lightgreen', 'lightcoral'])
    im = ax.imshow(mask_np, cmap=cmap, aspect='auto')
    
    # Add grid
    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks(np.arange(seq_len))
    ax.set_xticklabels([f'A{i}' for i in range(seq_len)])
    ax.set_yticklabels([f'A{i}' for i in range(seq_len)])
    
    # Add text annotations
    for i in range(seq_len):
        for j in range(seq_len):
            text = ax.text(j, i, '✓' if mask_np[i, j] == 0 else '✗',
                         ha="center", va="center", color="black", fontsize=14)
    
    ax.set_title("AR Decoder Causal Mask\n(Green=Attend, Red=Masked)", fontsize=14, weight='bold')
    ax.set_xlabel("Keys (previous tokens)", fontsize=12)
    ax.set_ylabel("Queries (current token)", fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Can Attend', 'Masked'])
    
    plt.tight_layout()
    return fig

def visualize_valle_combined_mask(text_len=4, audio_len=6):
    """Visualize the combined VALLE mask (text + audio)."""
    total_len = text_len + audio_len
    
    # Initialize mask
    mask = torch.zeros(total_len, total_len, dtype=torch.bool)
    
    # Text rows: can see all text, cannot see audio
    mask[:text_len, text_len:] = True
    
    # Audio rows: can see all text, causal for audio
    for i in range(text_len, total_len):
        # Causal mask for audio part
        mask[i, i+1:] = True
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    
    # Convert to numpy
    mask_np = mask.numpy().astype(int)
    
    # Plot
    cmap = ListedColormap(['lightgreen', 'lightcoral'])
    im = ax.imshow(mask_np, cmap=cmap, aspect='auto')
    
    # Labels
    labels = [f'T{i}' for i in range(text_len)] + [f'A{i}' for i in range(audio_len)]
    ax.set_xticks(np.arange(total_len))
    ax.set_yticks(np.arange(total_len))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(total_len):
        for j in range(total_len):
            text = ax.text(j, i, '✓' if mask_np[i, j] == 0 else '✗',
                         ha="center", va="center", color="black", fontsize=12)
    
    # Add separating lines
    ax.axhline(y=text_len-0.5, color='blue', linewidth=2)
    ax.axvline(x=text_len-0.5, color='blue', linewidth=2)
    
    ax.set_title("VALLE Combined Attention Mask (AR Stage)\n(Green=Attend, Red=Masked)", 
                 fontsize=14, weight='bold')
    ax.set_xlabel("Keys", fontsize=12)
    ax.set_ylabel("Queries", fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Can Attend', 'Masked'])
    
    # Add legend for regions
    text_patch = mpatches.Rectangle((0, 0), 1, 1, fc="lightyellow", edgecolor='blue', linewidth=2)
    audio_patch = mpatches.Rectangle((0, 0), 1, 1, fc="lightcyan", edgecolor='blue', linewidth=2)
    ax.legend([text_patch, audio_patch], ['Text Region', 'Audio Region'], 
              loc='upper left', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    return fig

def visualize_nar_mask(text_len=4, audio_len=6):
    """Visualize NAR decoder mask (no causal constraint in audio)."""
    total_len = text_len + audio_len
    
    # Initialize mask
    mask = torch.zeros(total_len, total_len, dtype=torch.bool)
    
    # Text rows: can see all text, cannot see audio
    mask[:text_len, text_len:] = True
    
    # Audio rows: can see all text AND all audio (bidirectional)
    # No causal constraint!
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    
    # Convert to numpy
    mask_np = mask.numpy().astype(int)
    
    # Plot
    cmap = ListedColormap(['lightgreen', 'lightcoral'])
    im = ax.imshow(mask_np, cmap=cmap, aspect='auto')
    
    # Labels
    labels = [f'T{i}' for i in range(text_len)] + [f'A{i}' for i in range(audio_len)]
    ax.set_xticks(np.arange(total_len))
    ax.set_yticks(np.arange(total_len))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(total_len):
        for j in range(total_len):
            text = ax.text(j, i, '✓' if mask_np[i, j] == 0 else '✗',
                         ha="center", va="center", color="black", fontsize=12)
    
    # Add separating lines
    ax.axhline(y=text_len-0.5, color='blue', linewidth=2)
    ax.axvline(x=text_len-0.5, color='blue', linewidth=2)
    
    ax.set_title("NAR Decoder Attention Mask\n(Green=Attend, Red=Masked) - Note: Audio is Bidirectional!", 
                 fontsize=14, weight='bold')
    ax.set_xlabel("Keys", fontsize=12)
    ax.set_ylabel("Queries", fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Can Attend', 'Masked'])
    
    plt.tight_layout()
    return fig

def visualize_padding_mask(text_len=6, text_valid=4, audio_len=8, audio_valid=5):
    """Visualize padding masks."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # Text padding mask
    text_mask = torch.zeros(text_len, dtype=torch.bool)
    text_mask[text_valid:] = True
    
    # Audio padding mask
    audio_mask = torch.zeros(audio_len, dtype=torch.bool)
    audio_mask[audio_valid:] = True
    
    # Plot text mask
    text_mask_np = text_mask.numpy().astype(int).reshape(1, -1)
    cmap = ListedColormap(['lightgreen', 'lightcoral'])
    
    im1 = axes[0].imshow(text_mask_np, cmap=cmap, aspect='auto')
    axes[0].set_yticks([0])
    axes[0].set_yticklabels(['Text'])
    axes[0].set_xticks(np.arange(text_len))
    axes[0].set_xticklabels([f'T{i}' for i in range(text_len)])
    axes[0].set_title('Text Padding Mask (x_mask)', fontsize=12, weight='bold')
    
    # Add annotations
    for i in range(text_len):
        label = 'Valid' if i < text_valid else 'PAD'
        color = 'black' if i < text_valid else 'darkred'
        axes[0].text(i, 0, label, ha="center", va="center", color=color, fontsize=10, weight='bold')
    
    # Plot audio mask
    audio_mask_np = audio_mask.numpy().astype(int).reshape(1, -1)
    im2 = axes[1].imshow(audio_mask_np, cmap=cmap, aspect='auto')
    axes[1].set_yticks([0])
    axes[1].set_yticklabels(['Audio'])
    axes[1].set_xticks(np.arange(audio_len))
    axes[1].set_xticklabels([f'A{i}' for i in range(audio_len)])
    axes[1].set_title('Audio Padding Mask (y_mask)', fontsize=12, weight='bold')
    
    # Add annotations
    for i in range(audio_len):
        label = 'Valid' if i < audio_valid else 'PAD'
        color = 'black' if i < audio_valid else 'darkred'
        axes[1].text(i, 0, label, ha="center", va="center", color=color, fontsize=10, weight='bold')
    
    plt.tight_layout()
    return fig

def create_all_visualizations():
    """Create all mask visualizations."""
    print("Generating VALLE mask visualizations...")
    
    # 1. AR Causal Mask
    fig1 = visualize_causal_mask(seq_len=8)
    fig1.savefig('/workspace/valle_mask_causal.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: valle_mask_causal.png")
    
    # 2. VALLE Combined Mask (AR)
    fig2 = visualize_valle_combined_mask(text_len=4, audio_len=6)
    fig2.savefig('/workspace/valle_mask_combined_ar.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: valle_mask_combined_ar.png")
    
    # 3. NAR Mask
    fig3 = visualize_nar_mask(text_len=4, audio_len=6)
    fig3.savefig('/workspace/valle_mask_nar.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: valle_mask_nar.png")
    
    # 4. Padding Masks
    fig4 = visualize_padding_mask(text_len=6, text_valid=4, audio_len=8, audio_valid=5)
    fig4.savefig('/workspace/valle_mask_padding.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: valle_mask_padding.png")
    
    plt.close('all')
    print("\nAll visualizations created successfully!")
    print("\nGenerated files:")
    print("  - valle_mask_causal.png       : AR decoder causal mask")
    print("  - valle_mask_combined_ar.png  : VALLE combined mask (AR stage)")
    print("  - valle_mask_nar.png          : NAR decoder mask (no causal)")
    print("  - valle_mask_padding.png      : Padding masks for text and audio")

def demonstrate_mask_logic():
    """Demonstrate mask creation logic with code."""
    print("\n" + "="*80)
    print("DEMONSTRATING MASK CREATION LOGIC")
    print("="*80)
    
    # Example dimensions
    text_len = 3
    audio_len = 4
    batch_size = 2
    
    print(f"\nExample: batch_size={batch_size}, text_len={text_len}, audio_len={audio_len}\n")
    
    # 1. Causal mask
    print("1. CAUSAL MASK (AR Decoder)")
    print("-" * 40)
    tgt_mask = torch.triu(torch.ones(audio_len, audio_len, dtype=torch.bool), diagonal=1)
    print("Code: torch.triu(torch.ones(audio_len, audio_len), diagonal=1)")
    print(f"Shape: {tgt_mask.shape}")
    print(f"Tensor:\n{tgt_mask.int()}")
    print("Interpretation: 1=masked (future), 0=can attend (past/present)\n")
    
    # 2. Padding mask
    print("2. PADDING MASK")
    print("-" * 40)
    text_lens = torch.tensor([2, 3])  # First batch has 2 valid tokens, second has 3
    from icefall.utils import make_pad_mask
    x_mask = make_pad_mask(text_lens)
    print(f"Code: make_pad_mask(text_lens) where text_lens={text_lens.tolist()}")
    print(f"Shape: {x_mask.shape}")
    print(f"Tensor:\n{x_mask.int()}")
    print("Interpretation: 1=padding (masked), 0=valid token\n")
    
    # 3. VALLE combined mask
    print("3. VALLE COMBINED MASK (Text + Audio)")
    print("-" * 40)
    total_len = text_len + audio_len
    mask = torch.zeros(total_len, total_len, dtype=torch.bool)
    mask[:text_len, text_len:] = True  # Text can't see audio
    for i in range(text_len, total_len):
        mask[i, i+1:] = True  # Audio causal
    
    print(f"Shape: {mask.shape}")
    print(f"Tensor:\n{mask.int()}")
    print("\nStructure:")
    print("  - Top-left: Text→Text (all zeros, bidirectional)")
    print("  - Top-right: Text→Audio (all ones, can't see audio)")
    print("  - Bottom-left: Audio→Text (all zeros, can see all text)")
    print("  - Bottom-right: Audio→Audio (upper triangular, causal)\n")
    
    # 4. Converting to additive mask
    print("4. CONVERTING TO ADDITIVE MASK")
    print("-" * 40)
    additive_mask = torch.zeros_like(mask, dtype=torch.float32)
    additive_mask.masked_fill_(mask, float('-inf'))
    print("Code: mask.masked_fill_(mask, float('-inf'))")
    print(f"Before (bool): {mask[0, :5].int().tolist()}")
    print(f"After (float): {additive_mask[0, :5].tolist()}")
    print("Interpretation: -inf will zero out attention weights after softmax\n")

if __name__ == "__main__":
    import sys
    
    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        print("Error: matplotlib is not installed. Please install it with:")
        print("  pip install matplotlib")
        sys.exit(1)
    
    # Create visualizations
    create_all_visualizations()
    
    # Demonstrate logic
    demonstrate_mask_logic()
    
    print("\n" + "="*80)
    print("For detailed explanations, see: VALLE_MASKS_ANALYSIS.md")
    print("="*80)
