# VALLE Masks: Quick Reference Guide

## TL;DR - The Three Main Mask Types

### 1. **Padding Masks** 🔒
- **What:** Mark which positions are padding (invalid)
- **Where:** `x_mask` (text), `y_mask` (audio)
- **Effect:** Prevents attention to padding, ensures only valid tokens contribute to loss
- **Value:** `True`=padding (masked), `False`=valid

### 2. **Causal Masks** ⏩
- **What:** Upper triangular matrix enforcing left-to-right generation
- **Where:** AR decoder only (`tgt_mask`)
- **Effect:** Each position can only see past positions, enables autoregressive generation
- **Value:** `True`=future (masked), `False`=past/present

### 3. **Combined Masks** 🔄
- **What:** Text bidirectional + Audio causal pattern
- **Where:** VALLE architecture (`xy_attn_mask`)
- **Effect:** Text can see all text, audio can see all text + past audio
- **Value:** Complex pattern (see visualization)

---

## Mask Cheat Sheet

| Component | Mask Used | Type | Shape | Purpose |
|-----------|-----------|------|-------|---------|
| **AR Decoder (VALLF)** | | | | |
| → Self-attention | `tgt_mask` | Causal | (L_audio, L_audio) | Left-to-right generation |
| → Target padding | `tgt_key_padding_mask` | Padding | (B, L_audio) | Ignore padded audio |
| → Cross-attention | None | None | - | All audio sees all text |
| → Memory padding | `memory_key_padding_mask` | Padding | (B, L_text) | Ignore padded text |
| **AR Decoder (VALLE)** | | | | |
| → Self-attention | `xy_attn_mask` | Combined | (L_text+L_audio, L_text+L_audio) | Text bi-dir + audio causal |
| → Source padding | Merged in `xy_attn_mask` | Padding | - | Ignore all padding |
| **NAR Decoder** | | | | |
| → Self-attention | None | None | - | Bidirectional (parallel) |
| → Source padding | `xy_padding_mask` | Padding | (B, L_text+L_audio) | Ignore padded positions |

---

## Code Patterns

### Creating Causal Mask
```python
# Upper triangular mask for autoregressive generation
tgt_mask = torch.triu(
    torch.ones(seq_len, seq_len, dtype=torch.bool),
    diagonal=1  # Start masking one position to the right
)
# Result: position i can see positions 0 to i (inclusive)
```

### Creating Padding Mask
```python
# From sequence lengths
from icefall.utils import make_pad_mask
x_mask = make_pad_mask(x_lens).to(device)
# x_lens = [3, 2] → x_mask = [[0,0,0], [0,0,1]] (for max_len=3)
```

### Creating Combined VALLE Mask
```python
# Step 1: Text region (can see text, not audio)
x_attn_mask = F.pad(
    torch.zeros((x_len, x_len), dtype=torch.bool),
    (0, y_len),  # Pad on the right (audio side)
    value=True   # Mask audio
)

# Step 2: Audio region (can see all text, causal audio)
y_attn_mask = F.pad(
    torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
    (x_len, 0),  # Pad on the left (text side)
    value=False  # Can see text
)

# Step 3: Combine
xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)

# Step 4: Merge with padding
xy_attn_mask = xy_attn_mask.logical_or(padding_mask_expanded)

# Step 5: Convert to additive (-inf for masked)
xy_attn_mask = torch.zeros_like(xy_attn_mask, dtype=torch.float)
xy_attn_mask.masked_fill_(xy_attn_mask, float('-inf'))
```

### Using Masks in Attention
```python
# VALLF AR Decoder (separate text and audio)
y_dec, _ = self.ar_decoder(
    tgt=y_pos,                          # Audio queries
    memory=x,                           # Text keys/values
    tgt_mask=tgt_mask,                  # Causal mask
    tgt_key_padding_mask=ar_y_mask,    # Audio padding
    memory_key_padding_mask=x_mask,     # Text padding
)

# VALLE AR Decoder (concatenated)
xy_dec, _ = self.ar_decoder(
    src=xy_pos,                         # Combined text+audio
    mask=xy_attn_mask,                  # Combined causal+padding
)

# NAR Decoder (no causal mask)
xy_dec, _ = self.nar_decoder(
    src=xy_pos,
    src_key_padding_mask=xy_padding_mask,  # Only padding mask
)
```

---

## Visual Patterns

### AR Causal Mask (8 positions)
```
Keys → 0  1  2  3  4  5  6  7
     ┌──────────────────────┐
Q  0 │ ✓  ✗  ✗  ✗  ✗  ✗  ✗  ✗ │
u  1 │ ✓  ✓  ✗  ✗  ✗  ✗  ✗  ✗ │
e  2 │ ✓  ✓  ✓  ✗  ✗  ✗  ✗  ✗ │
r  3 │ ✓  ✓  ✓  ✓  ✗  ✗  ✗  ✗ │
i  4 │ ✓  ✓  ✓  ✓  ✓  ✗  ✗  ✗ │
e  5 │ ✓  ✓  ✓  ✓  ✓  ✓  ✗  ✗ │
s  6 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✗ │
   7 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓ │
     └──────────────────────┘
```

### VALLE Combined Mask (3 text + 4 audio)
```
Keys → T0 T1 T2 A0 A1 A2 A3
     ┌─────────────────────┐
Q T0 │ ✓  ✓  ✓  ✗  ✗  ✗  ✗ │ ← Text sees all text
u T1 │ ✓  ✓  ✓  ✗  ✗  ✗  ✗ │   (bidirectional)
e T2 │ ✓  ✓  ✓  ✗  ✗  ✗  ✗ │
r ───┼─────────────────────┤
i A0 │ ✓  ✓  ✓  ✓  ✗  ✗  ✗ │ ← Audio sees all text
e A1 │ ✓  ✓  ✓  ✓  ✓  ✗  ✗ │   + causal audio
s A2 │ ✓  ✓  ✓  ✓  ✓  ✓  ✗ │
  A3 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓ │
     └─────────────────────┘
```

### NAR Mask (3 text + 4 audio)
```
Keys → T0 T1 T2 A0 A1 A2 A3
     ┌─────────────────────┐
Q T0 │ ✓  ✓  ✓  ✗  ✗  ✗  ✗ │ ← Text sees all text
u T1 │ ✓  ✓  ✓  ✗  ✗  ✗  ✗ │
e T2 │ ✓  ✓  ✓  ✗  ✗  ✗  ✗ │
r ───┼─────────────────────┤
i A0 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓ │ ← Audio sees all text
e A1 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓ │   + ALL audio (bidirectional!)
s A2 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓ │
  A3 │ ✓  ✓  ✓  ✓  ✓  ✓  ✓ │
     └─────────────────────┘
```

---

## Common Patterns & Pitfalls

### ✅ Correct Usage

```python
# 1. Always mask padding in loss computation
targets = codes[..., stage] + NUM_AUDIO_TOKENS * y_mask_int
loss = F.cross_entropy(logits, targets, ignore_index=NUM_AUDIO_TOKENS)

# 2. Adjust mask when prepending BOS
if self.ar_audio_prepend_bos:
    ar_y_mask = F.pad(y_mask, (1, 0), value=False)

# 3. Convert boolean to additive for attention
attn_mask_additive = torch.zeros_like(attn_mask, dtype=torch.float)
attn_mask_additive.masked_fill_(attn_mask, float('-inf'))
```

### ❌ Common Mistakes

```python
# 1. Wrong: Using attention mask without merging padding
xy_dec = decoder(src, mask=xy_attn_mask)  # Missing padding!

# 2. Wrong: Forgetting to mask loss on padding
loss = F.cross_entropy(logits, targets)  # No ignore_index!

# 3. Wrong: Using causal mask in NAR
nar_dec = nar_decoder(src, mask=tgt_mask)  # NAR should be bidirectional!

# 4. Wrong: Boolean mask without conversion
attn = attention(q, k, v, mask=bool_mask)  # Should be additive!
```

---

## Mask Control Parameters

### Model Initialization
```python
VALLE(
    prepend_bos=False,    # Add BOS token? (adds 1 to sequence length)
    prefix_mode=1,        # 0=no prompt, 1=prefix, 2=random, 4=external
    num_quantizers=8,     # Affects NAR stages (1 for AR, rest for NAR)
)
```

### Training
```python
model.forward(
    train_stage=0,  # 0=AR+NAR, 1=AR only, 2=NAR only
)
```

### Inference
```python
model.inference(
    top_k=-100,         # Sampling parameter
    temperature=1.0,    # Sampling temperature
)
```

---

## Debugging Tips

### Check Mask Shape
```python
print(f"x_mask shape: {x_mask.shape}")        # Should be (B, L_text)
print(f"y_mask shape: {y_mask.shape}")        # Should be (B, L_audio)
print(f"tgt_mask shape: {tgt_mask.shape}")    # Should be (L_audio, L_audio)
print(f"xy_attn_mask shape: {xy_attn_mask.shape}")  # Should be (L_text+L_audio, L_text+L_audio)
```

### Visualize Mask
```python
import matplotlib.pyplot as plt
plt.imshow(mask.cpu().numpy(), cmap='RdYlGn_r')
plt.colorbar()
plt.title("Attention Mask")
plt.show()
```

### Check Mask Values
```python
# Causal mask should be upper triangular
assert torch.all(tgt_mask == torch.triu(tgt_mask, diagonal=1))

# Padding mask should only have True for padded positions
assert torch.all(y_mask[:, :y_lens.min()] == False)

# Combined mask should have correct structure
assert torch.all(xy_attn_mask[:text_len, :text_len] == False)  # Text bi-dir
assert torch.all(xy_attn_mask[:text_len, text_len:] == True)   # Text can't see audio
```

### Verify Loss Computation
```python
# Check that padding is ignored
valid_loss = loss * (1 - y_mask.float())
print(f"Loss on padding: {loss[y_mask].sum()}")  # Should be ~0
print(f"Loss on valid tokens: {valid_loss.sum()}")
```

---

## Performance Impact

| Mask Type | Training Cost | Inference Cost | Quality Impact |
|-----------|---------------|----------------|----------------|
| **No masks** | ❌ Unstable | ❌ Incorrect | ❌ Poor |
| **Padding only** | ⚠️ Works but suboptimal | ⚠️ Unrealistic | ⚠️ Degraded |
| **Causal only (AR)** | ✅ Good | ✅ Correct | ✅ High quality |
| **All masks (AR+NAR)** | ✅ Optimal | ✅ Fast & correct | ✅ Best quality |

---

## Key Takeaways

1. **Padding masks** are essential for variable-length sequences
   - Without them: model learns from padding, unstable training
   - With them: only valid tokens contribute to learning

2. **Causal masks** enable autoregressive generation
   - Without them: model cheats by seeing future
   - With them: training matches inference behavior

3. **Combined masks** optimize VALLE architecture
   - Text: bidirectional (efficient encoding)
   - Audio: causal (proper generation)
   - Result: best of both worlds

4. **NAR uses no causal mask**
   - Predicts all positions in parallel
   - 8x faster than fully autoregressive
   - Bidirectional context improves quality

5. **Mask merging** combines multiple constraints
   - Attention mask + padding mask → final mask
   - Boolean → additive conversion for attention
   - Careful broadcasting for multi-head attention

---

## Further Reading

- **Full Analysis:** See `VALLE_MASKS_ANALYSIS.md` for detailed explanations
- **VALLE Paper:** [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111)
- **Transformer Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Implementation:** `/workspace/valle/models/valle.py`

---

*Last updated: 2025-11-20*
