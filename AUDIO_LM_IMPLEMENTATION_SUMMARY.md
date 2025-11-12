# AudioLM Implementation Summary

## Overview

I've created a new audio-to-audio generation model called **AudioLM** based on VALLE's architecture. The key modification is replacing text/MIDI conditioning with audio conditioning while keeping the AR+NAR structure.

## Files Created/Modified

### 1. **New Model File**: `/workspace/valle/models/audio_lm.py`

Complete implementation of AudioLM with:
- AR decoder for 1st quantizer prediction
- NAR decoder for remaining quantizers (2-4)
- Audio-based conditioning instead of text/MIDI

**Key Classes**:
- `AudioLM`: Main model class with AR+NAR architecture

### 2. **Modified**: `/workspace/valle/models/__init__.py`

Added AudioLM import and model initialization:
```python
from .audio_lm import AudioLM

# In get_model():
elif params.model_name.lower() in ["audio-lm", "audiolm"]:
    model = AudioLM(...)
```

### 3. **New Training Script**: `/workspace/egs/atepp/bin/trainer_audiolm.py`

Custom training script for AudioLM with:
- `compute_loss_audiolm()`: Handles audio splitting (prompt + target)
- `train_one_epoch_audiolm()`: Modified training loop
- `run_audiolm()`: Main training function

### 4. **Documentation**: `/workspace/valle/models/AUDIO_LM_GUIDE.md`

Comprehensive guide covering:
- Model architecture
- Usage examples
- Training commands
- Data preparation
- Troubleshooting

## Architecture Comparison

### VALLE (Original)
```
Text/MIDI Tokens (conditioning)
        ↓
    [AR Decoder]
        ↓
Audio Tokens (1st quantizer)
        ↓
    [NAR Decoder]
        ↓
Audio Tokens (2-8 quantizers)
```

### AudioLM (New)
```
Prompt Audio Tokens (conditioning)
        ↓
    [AR Decoder]
        ↓
Target Audio Tokens (1st quantizer)
        ↓
    [NAR Decoder]
        ↓
Target Audio Tokens (2-4 quantizers)
```

## Key Modifications

### 1. Input Changes

**VALLE**:
- `x`: Text/MIDI tokens (1D or 2D)
- `y`: Audio tokens (3D: batch × time × quantizers)

**AudioLM**:
- `x`: Prompt audio tokens (3D: batch × time × quantizers)
- `y`: Target audio tokens (3D: batch × time × quantizers)

### 2. Embedding Layers

**VALLE**:
```python
self.ar_text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)
```

**AudioLM**:
```python
# Embed all quantizers of prompt audio
self.ar_prompt_embeddings = nn.ModuleList([
    TokenEmbedding(d_model, NUM_AUDIO_TOKENS)
    for _ in range(num_quantizers)
])
```

### 3. Forward Pass

**VALLE**: Concatenates text embeddings + audio embeddings
```python
xy_pos = torch.concat([text_embeddings, audio_embeddings], dim=1)
```

**AudioLM**: Concatenates prompt audio embeddings + target audio embeddings
```python
xy_pos = torch.concat([prompt_embeddings, target_embeddings], dim=1)
```

### 4. Training Data Splitting

AudioLM automatically splits each audio sequence during training:
```python
# Split audio: 40% prompt, 60% target
split_point = (audio_len * 0.4).long()
x = audio[:, :split_point, :]  # Prompt
y = audio[:, split_point:, :]  # Target
```

## Usage

### Training Command

```bash
python3 bin/trainer_audiolm.py \
    --model-name audio-lm \
    --decoder-dim 1024 \
    --nhead 16 \
    --num-decoder-layers 12 \
    --num-quantizers 4 \
    --base-lr 0.05 \
    --warmup-steps 200 \
    --num-epochs 20 \
    --exp-dir exp/audio-lm \
    --max-duration 150 \
    --filter-min-duration 5 \
    --filter-max-duration 15 \
    --train-stage 0 \
    --manifest-dir data/tokenized
```

### Inference Example

```python
from valle.models import AudioLM

# Initialize model
model = AudioLM(
    d_model=1024,
    nhead=16,
    num_layers=12,
    num_quantizers=4,
)

# Load checkpoint
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

# Generate continuation
prompt_tokens = ...  # (1, 150, 4) - 3 seconds
prompt_lens = torch.tensor([150])

generated = model.inference(
    x=prompt_tokens,
    x_lens=prompt_lens,
    max_new_tokens=500,
    temperature=1.0,
)
```

## Advantages

1. **Simpler Input**: No need for text/MIDI preprocessing
2. **Direct Audio Control**: Conditioning directly on audio features
3. **Flexible Applications**: Audio continuation, in-filling, style transfer
4. **Same Training Paradigm**: Reuses VALLE's proven AR+NAR structure
5. **Efficient**: Inherits VALLE's efficient multi-stage generation

## Technical Details

### Attention Masks

**AR Stage** (Causal):
```
Prompt: Can attend to all prompt tokens
Target: Can attend to prompt + previous target tokens

    P1 P2 P3 | T1 T2 T3
P1  ✓  ✓  ✓  | ✗  ✗  ✗
P2  ✓  ✓  ✓  | ✗  ✗  ✗
P3  ✓  ✓  ✓  | ✗  ✗  ✗
----|--------|---------
T1  ✓  ✓  ✓  | ✓  ✗  ✗
T2  ✓  ✓  ✓  | ✓  ✓  ✗
T3  ✓  ✓  ✓  | ✓  ✓  ✓
```

**NAR Stage** (Non-Causal):
```
Prompt: Can attend to all prompt
Target: Can attend to prompt + all target

    P1 P2 P3 | T1 T2 T3
P1  ✓  ✓  ✓  | ✗  ✗  ✗
P2  ✓  ✓  ✓  | ✗  ✗  ✗
P3  ✓  ✓  ✓  | ✗  ✗  ✗
----|--------|---------
T1  ✓  ✓  ✓  | ✓  ✓  ✓
T2  ✓  ✓  ✓  | ✓  ✓  ✓
T3  ✓  ✓  ✓  | ✓  ✓  ✓
```

### Loss Computation

```python
# AR Loss: Cross-entropy for 1st quantizer
ar_loss = CrossEntropy(predicted_q1, target_q1)

# NAR Loss: Cross-entropy for remaining quantizers
# One stage is randomly selected during training
nar_stage = random.choice([1, 2, 3])  # Which quantizer to predict
nar_loss = CrossEntropy(predicted_q[nar_stage], target_q[nar_stage])

# Total loss
total_loss = (ar_loss + nar_loss) / 2
```

### Quantizer Hierarchy

Following VALLE's design:
- **Quantizer 1** (AR): Coarse acoustic structure
- **Quantizer 2-4** (NAR): Fine-grained acoustic details

Each quantizer adds progressively finer details to the audio.

## Compatibility

### Data Format
AudioLM uses the **same data format** as VALLE:
- EnCodec 32kHz tokenized audio
- 4 quantizers per frame
- 50Hz frame rate
- Standard lhotse manifests

### Training Infrastructure
AudioLM is compatible with:
- Existing data loaders (`TtsDataModule`)
- Dynamic batching (`DynamicBucketingSampler`)
- Training utilities (checkpointing, logging, metrics)
- Multi-GPU training (DDP)

### Model Components
Reuses VALLE's modules:
- `TokenEmbedding`
- `SinePositionalEmbedding`
- `TransformerEncoder`
- `TransformerEncoderLayer`
- `AdaptiveLayerNorm`

## Future Enhancements

Possible improvements:
1. **Variable split ratios**: Dynamic prompt/target splitting
2. **Prefix modes**: Different conditioning strategies (like VALLE's prefix_mode)
3. **Multi-stage AR**: Predict multiple quantizers autoregressively
4. **Conditional generation**: Add style tokens or speaker embeddings
5. **Efficient caching**: KV-cache for faster inference

## Testing Checklist

Before training:
- [ ] Verify audio tokenization (4 quantizers, 50Hz)
- [ ] Check data duration (5-15 seconds recommended)
- [ ] Confirm manifest format matches VALLE
- [ ] Test forward pass with dummy data
- [ ] Validate loss computation

During training:
- [ ] Monitor AR accuracy (should reach >50%)
- [ ] Monitor NAR accuracy (should reach >40%)
- [ ] Check for loss convergence
- [ ] Verify gradient flow (no NaN/Inf)
- [ ] Sample generation quality

## Differences from Standard Language Models

| Aspect | Text LLM | AudioLM |
|--------|----------|---------|
| Input | Text tokens | Audio tokens (multi-level) |
| Conditioning | Previous tokens only | Prompt audio + previous tokens |
| Output | 1 token/step | 1 token (AR) + 3 tokens (NAR) |
| Architecture | Decoder-only | AR Decoder + NAR Decoder |
| Loss | Single CE loss | AR CE + NAR CE |
| Generation | Purely sequential | Sequential (AR) + Parallel (NAR) |

## Summary

AudioLM successfully adapts VALLE's architecture for audio-to-audio generation by:
1. Replacing text/MIDI embeddings with audio embeddings
2. Using audio prompts as conditioning
3. Maintaining AR+NAR structure for efficiency
4. Reusing proven training infrastructure

The model is ready for training on audio continuation tasks and can be easily extended for other audio generation applications.
