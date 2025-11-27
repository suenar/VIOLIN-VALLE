# Audio-to-Audio Inference Guide

This guide explains how to use the VALLE-audio model for audio-to-audio generation (audio continuation).

## Overview

The VALLE-audio model takes an audio prompt (e.g., 3-5 seconds) and generates a continuation of that audio. This is useful for:
- Audio continuation/completion
- Music generation from a short prompt
- Extending audio samples

## Quick Start

### Basic Usage

```bash
python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts ./prompts/prompt_A.wav \
    --output-dir infer/audio_continuation \
    --output-file generated
```

This will:
1. Load the trained VALLE-audio model
2. Use `prompt_A.wav` as the conditioning audio
3. Generate a continuation (~10 seconds by default)
4. Save two files:
   - `generated_0_full.wav` - Full audio (prompt + generated continuation)
   - `generated_0_continuation.wav` - Only the generated continuation

### Advanced Usage

```bash
python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts ./prompts/prompt_A.wav \
    --output-dir infer/audio_continuation \
    --output-file generated \
    --max-duration 15.0 \
    --top-k 50 \
    --temperature 0.8 \
    --audio-tokenizer Encodec32Violin
```

### Multiple Prompts

You can process multiple audio prompts in one run:

```bash
python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts "./prompts/prompt_A.wav|./prompts/prompt_B.wav" \
    --output-dir infer/audio_continuation \
    --output-file generated
```

This will generate:
- `generated_0_full.wav`, `generated_0_continuation.wav` (from prompt_A.wav)
- `generated_1_full.wav`, `generated_1_continuation.wav` (from prompt_B.wav)

## Command Line Arguments

### Required Arguments

- `--checkpoint`: Path to the trained model checkpoint
- `--audio-prompts`: Audio prompt file(s), separated by `|` for multiple prompts

### Optional Arguments

#### Output Settings
- `--output-dir`: Directory to save generated files (default: `infer/audio_continuation`)
- `--output-file`: Base name for output files (default: `generated`)

#### Generation Parameters
- `--max-duration`: Maximum duration to generate in seconds (default: `10.0`)
- `--top-k`: Top-k sampling parameter (default: `-100` for unrestricted)
  - Use positive values (e.g., `50`) for more conservative/coherent generation
  - Use `-100` or large negative values for more diverse generation
- `--temperature`: Sampling temperature (default: `1.0`)
  - Lower values (e.g., `0.7`) make generation more conservative
  - Higher values (e.g., `1.2`) make generation more random

#### Model Configuration
- `--model-name`: Model name (default: `VALLE-audio`)
- `--decoder-dim`: Model dimension (default: `1024`)
- `--nhead`: Number of attention heads (default: `16`)
- `--num-decoder-layers`: Number of decoder layers (default: `12`)
- `--num-quantizers`: Number of audio quantizers (default: `4`)
- `--audio-tokenizer`: Which tokenizer to use (default: `Encodec32Violin`)
  - Options: `Encodec`, `Encodec32`, `Encodec32FT`, `Encodec32Violin`

## Model Architecture

The VALLE-audio model uses a two-stage generation process:

1. **AR (AutoRegressive) Stage**: Generates the first quantizer autoregressively
   - Takes the audio prompt as conditioning
   - Predicts audio tokens one-by-one until EOS or max length
   
2. **NAR (Non-AutoRegressive) Stage**: Generates remaining quantizers in parallel
   - Takes the first quantizer as input
   - Predicts quantizers 2-4 simultaneously

## Tips for Better Generation

### Prompt Quality
- Use clean, high-quality audio prompts
- Prompts should be 3-5 seconds long for best results
- The model will try to continue in a similar style to the prompt

### Sampling Parameters

**Conservative Generation** (more coherent, less diverse):
```bash
--top-k 30 --temperature 0.7
```

**Balanced Generation** (default):
```bash
--top-k -100 --temperature 1.0
```

**Creative Generation** (more diverse, potentially less coherent):
```bash
--top-k -100 --temperature 1.2
```

## Output Files

The inference script generates two files per prompt:

1. **`{output_file}_{idx}_full.wav`**
   - Contains: Prompt audio + Generated continuation
   - Useful for: Hearing the complete sequence
   
2. **`{output_file}_{idx}_continuation.wav`**
   - Contains: Only the generated continuation (no prompt)
   - Useful for: Analyzing what the model generated

## Programmatic Usage

You can also use the model programmatically:

```python
import torch
from valle.models.valle_audio import VALLE_Audio
from valle.data import tokenize_audio, AudioTokenizer32Violin

# Load model
model = VALLE_Audio(
    d_model=1024,
    nhead=16,
    num_layers=12,
    num_quantizers=4,
)
model.load_state_dict(torch.load("checkpoint.pt")["model"])
model.eval()
model.cuda()

# Tokenize audio prompt
tokenizer = AudioTokenizer32Violin()
audio_tokens = tokenize_audio("Encodec32Violin", "prompt.wav", tokenizer)
audio_tokens = audio_tokens.cuda()
audio_lens = torch.tensor([audio_tokens.shape[1]], device="cuda")

# Generate continuation
with torch.no_grad():
    generated = model.inference(
        x=audio_tokens,
        x_lens=audio_lens,
        max_new_tokens=750,  # ~10 seconds at 75 fps
        top_k=-100,
        temperature=1.0,
        return_full_sequence=True,  # Include prompt in output
    )

# Decode to audio
audio_samples = tokenizer.decode(generated.transpose(1, 2), None)

# Save
import torchaudio
torchaudio.save("generated.wav", audio_samples[0].cpu(), 32000)
```

## Troubleshooting

### Issue: Out of memory
**Solution**: Reduce `--max-duration` or use a smaller model

### Issue: Generated audio is too repetitive
**Solution**: Increase `--temperature` or use positive `--top-k` value

### Issue: Generated audio is too noisy/incoherent
**Solution**: Decrease `--temperature` and/or use smaller `--top-k` value

### Issue: Checkpoint loading fails
**Solution**: Ensure model configuration matches training configuration:
```bash
--decoder-dim 1024 \
--nhead 16 \
--num-decoder-layers 12 \
--num-quantizers 4
```

## Differences from Original VALLE

The VALLE-audio model differs from the original VALLE in the following ways:

1. **Input**: Takes audio tokens instead of text/MIDI tokens
2. **Conditioning**: Uses audio prompt instead of text/MIDI prompt
3. **Task**: Audio continuation instead of text-to-speech/MIDI-to-audio
4. **Inference**: No need for text/MIDI tokenization

The architecture (AR + NAR) remains the same, making it easy to understand if you're familiar with VALLE.

## Example Workflow

```bash
# 1. Train the model (see training guide)
python3 bin/trainer_audiolm.py \
    --model-name VALLE-audio \
    --exp-dir exp/VALLE-audio \
    ...

# 2. Prepare audio prompts (3-5 second clips)
# Place prompts in ./prompts/ directory

# 3. Run inference
python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts ./prompts/my_prompt.wav \
    --output-dir results \
    --output-file test

# 4. Listen to results
# results/test_0_full.wav - Full sequence
# results/test_0_continuation.wav - Generated part only
```

## References

- Original VALLE paper: [arXiv:2301.02111](https://arxiv.org/abs/2301.02111)
- EnCodec audio codec: [arXiv:2210.13438](https://arxiv.org/abs/2210.13438)
