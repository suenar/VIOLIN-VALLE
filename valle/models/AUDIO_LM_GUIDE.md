# AudioLM Model Guide

## Overview

AudioLM is an audio-to-audio generation model based on the VALLE architecture. Unlike VALLE which uses text/MIDI as conditioning, AudioLM takes **audio tokens only** as input and generates audio continuations.

### Key Features

- **Input**: Audio tokens (3-10 seconds prompt)
- **Output**: Audio tokens (generated continuation)
- **Architecture**: AR (AutoRegressive) + NAR (Non-AutoRegressive)
  - **AR Stage**: Predicts 1st quantizer autoregressively
  - **NAR Stage**: Predicts remaining quantizers (2-4) in parallel
- **Loss**: Cross-entropy with masking (same as LLMs)

## Architecture Details

### Model Structure

```
Input: Prompt Audio Tokens (B, S, 4)
       ↓
   [AR Decoder]
       ↓
   1st Quantizer Prediction (B, T, 1)
       ↓
   [NAR Decoder]
       ↓
   Remaining Quantizers (B, T, 3)
       ↓
Output: Full Audio Tokens (B, T, 4)
```

### Compared to VALLE

| Component | VALLE | AudioLM |
|-----------|-------|---------|
| Conditioning Input | Text/MIDI tokens | Audio tokens (prompt) |
| Target Output | Audio tokens | Audio tokens (continuation) |
| AR Stage | Predicts 1st quantizer from text | Predicts 1st quantizer from audio |
| NAR Stage | Predicts 2-8 quantizers | Predicts 2-4 quantizers |
| Use Case | Text-to-Speech, MIDI-to-Audio | Audio continuation, in-painting |

## Usage

### 1. Model Initialization

```python
from valle.models import AudioLM

model = AudioLM(
    d_model=1024,           # Model dimension
    nhead=16,               # Number of attention heads
    num_layers=12,          # Number of transformer layers
    norm_first=True,        # Pre-normalization
    add_prenet=False,       # Add PreNet layers
    num_quantizers=4,       # Number of audio quantizers
    nar_scale_factor=1.0,   # NAR decoder scale
    prepend_bos=False,      # Prepend BOS token
)
```

### 2. Training

The model expects audio sequences to be split into:
- **Prompt (x)**: First 30-40% of the audio (conditioning)
- **Target (y)**: Remaining 60-70% (what to predict)

```python
# Example training forward pass
x = prompt_audio_tokens  # (B, S, 4) - e.g., first 3 seconds
y = target_audio_tokens  # (B, T, 4) - e.g., next 7 seconds
x_lens = torch.tensor([150])  # 3 seconds at 50Hz = 150 frames
y_lens = torch.tensor([350])  # 7 seconds at 50Hz = 350 frames

predictions, loss, metrics = model(
    x=x,
    x_lens=x_lens,
    y=y,
    y_lens=y_lens,
    train_stage=0,  # 0=both AR+NAR, 1=AR only, 2=NAR only
)
```

### 3. Training Script

Use the provided training script:

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

**Note**: Adjust `--filter-min-duration` and `--filter-max-duration` to control the length of audio samples (5-15 seconds recommended).

### 4. Inference

```python
# Load model
model = AudioLM(...)
model.load_state_dict(checkpoint)
model.eval()

# Prepare prompt audio
prompt_tokens = ...  # (1, S, 4) - e.g., 3 seconds of audio
prompt_lens = torch.tensor([150])

# Generate continuation
generated = model.inference(
    x=prompt_tokens,
    x_lens=prompt_lens,
    max_new_tokens=500,    # Maximum tokens to generate
    top_k=-100,            # Top-k sampling
    temperature=1.0,       # Sampling temperature
)

# generated shape: (1, S+T, 4)
```

## Data Preparation

### Input Requirements

1. **Audio Format**: Audio should be tokenized using EnCodec 32kHz (4 quantizers)
2. **Duration**: 5-15 seconds per sample (can be split during training)
3. **Frame Rate**: 50Hz (640 samples stride at 32kHz)

### Splitting Strategy

During training, each audio sample is split into:
- **Prompt**: 30-40% of the audio (minimum 50 frames, ~1 second)
- **Target**: Remaining 60-70%

You can modify the split ratio in `compute_loss_audiolm()`:

```python
# In trainer_audiolm.py, line ~69
split_points = (audio_features_lens * 0.4).long()  # 40% prompt, 60% target
```

### Data Format

The model expects the same data format as VALLE:

```
data/tokenized/
├── cuts_train.jsonl.gz
├── cuts_validation.jsonl.gz
└── cuts_test.jsonl.gz
```

Each cut should have `audio_features` with shape `(T, 4)` where:
- `T`: Number of time frames
- `4`: Number of quantizers

## Training Stages

Like VALLE, AudioLM supports staged training:

### Stage 0: Joint Training (Default)
```bash
--train-stage 0  # Train both AR and NAR together
```

### Stage 1: AR Only
```bash
--train-stage 1  # Train only AR decoder (1st quantizer)
```

### Stage 2: NAR Only
```bash
--train-stage 2  # Train only NAR decoder (2-4 quantizers)
# Requires pretrained AR stage
```

## Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 1024 | Model dimension |
| `nhead` | 16 | Number of attention heads |
| `num_layers` | 12 | Number of transformer layers |
| `num_quantizers` | 4 | Number of audio quantizers (EnCodec 32kHz) |
| `nar_scale_factor` | 1.0 | NAR decoder dimension scale |
| `prepend_bos` | False | Add BOS token for AR |
| `add_prenet` | False | Add PreNet layers |

## Use Cases

1. **Audio Continuation**: Generate continuation of audio (e.g., music generation)
2. **Audio In-filling**: Fill gaps in audio sequences
3. **Audio Style Transfer**: Continue audio in the style of the prompt
4. **Unconditional Generation**: Generate audio from silence/noise prompts

## Performance Tips

1. **Longer prompts**: Use 3-5 seconds for better conditioning
2. **Variable split**: Randomly vary the prompt/target split during training
3. **Temperature**: Lower temperature (0.8-0.9) for more stable generation
4. **Max duration**: Use `--max-duration 150` to balance batch size and memory
5. **Staged training**: Train AR first, then NAR for better convergence

## Differences from Standard LLM

While AudioLM uses autoregressive generation similar to LLMs, there are key differences:

| Aspect | Text LLM | AudioLM |
|--------|----------|---------|
| Tokens | Discrete words | Audio codec tokens (4 layers) |
| Conditioning | Previous text | Prompt audio + previous tokens |
| Generation | Single token per step | 1 token (AR) + 3 tokens (NAR) |
| Sequence Length | 100s-1000s | 100s-1000s frames (~2-20 seconds) |
| Loss | Cross-entropy | Cross-entropy (AR) + Cross-entropy (NAR) |

## Troubleshooting

### Issue: Model generates repetitive audio
**Solution**: Increase temperature, use top-k/top-p sampling, or increase model capacity

### Issue: Poor audio quality
**Solution**: 
- Ensure NAR stage is trained properly
- Check that all 4 quantizers are being predicted
- Verify audio tokenization is correct

### Issue: Slow inference
**Solution**: 
- Reduce `max_new_tokens`
- Use smaller model (reduce `d_model`, `num_layers`)
- Consider caching KV in transformer

### Issue: OOM during training
**Solution**: 
- Reduce `--max-duration` (e.g., from 200 to 100)
- Reduce `--decoder-dim` (e.g., from 1024 to 768)
- Use gradient accumulation `--accumulate-grad-steps`

## Citation

If you use this model, please cite the original VALLE paper:

```bibtex
@article{wang2023valle,
  title={Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers},
  author={Wang, Chengyi and Chen, Sanyuan and Wu, Yu and others},
  journal={arXiv preprint arXiv:2301.02111},
  year={2023}
}
```
