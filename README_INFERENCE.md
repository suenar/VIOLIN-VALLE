# VALLE-Audio Inference System

Complete inference system for audio-to-audio generation using the VALLE-audio model.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [What's Included](#whats-included)
3. [Installation Check](#installation-check)
4. [Usage Examples](#usage-examples)
5. [Documentation](#documentation)
6. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### 1. Verify Setup

```bash
cd egs/atepp
python3 bin/verify_inference_setup.py
```

If you have a checkpoint:
```bash
python3 bin/verify_inference_setup.py --checkpoint exp/VALLE-audio/checkpoint-20000.pt
```

### 2. Generate Audio

**Option A: Using the wrapper script (recommended)**
```bash
./bin/generate_audio.sh \
    -c exp/VALLE-audio/checkpoint-20000.pt \
    -p prompts/prompt_A.wav
```

**Option B: Using the Python script directly**
```bash
python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts prompts/prompt_A.wav \
    --output-dir results
```

### 3. Check Results

```bash
ls results/
# generated_0_full.wav - Full audio (prompt + continuation)
# generated_0_continuation.wav - Generated part only
```

## 📦 What's Included

### Scripts

| File | Purpose | Type |
|------|---------|------|
| `bin/infer_audiolm.py` | Main inference script | Python |
| `bin/generate_audio.sh` | Easy-to-use wrapper | Bash |
| `bin/verify_inference_setup.py` | Setup verification | Python |

### Documentation

| File | Description |
|------|-------------|
| `AUDIO_TO_AUDIO_INFERENCE.md` | Complete user guide with examples |
| `QUICK_REFERENCE_INFERENCE.md` | Quick reference for common tasks |
| `INFERENCE_SETUP_COMPLETE.md` | Technical setup details |
| `CHANGES_SUMMARY.md` | Summary of all changes made |
| `README_INFERENCE.md` | This file |

### Model Enhancement

- **Modified**: `valle/models/valle_audio.py`
  - Enhanced `inference()` method with `return_full_sequence` parameter
  - Better support for audio continuation use cases

## ✅ Installation Check

Run the verification script to ensure everything is set up correctly:

```bash
cd egs/atepp
python3 bin/verify_inference_setup.py
```

Expected output:
```
✓ Main inference script: bin/infer_audiolm.py
✓ Wrapper script: bin/generate_audio.sh
✓ Main inference script is executable
✓ Wrapper script is executable
✓ VALLE-audio model: ../../valle/models/valle_audio.py
✓ Package 'torch' is installed
✓ Package 'torchaudio' is installed
...
✓ All checks passed! ✨
```

## 💡 Usage Examples

### Example 1: Basic Audio Continuation

Generate a 10-second continuation from a prompt:

```bash
./bin/generate_audio.sh \
    -c exp/VALLE-audio/checkpoint-20000.pt \
    -p prompts/my_music.wav
```

### Example 2: Long Generation

Generate a 20-second continuation:

```bash
./bin/generate_audio.sh \
    -c exp/VALLE-audio/checkpoint-20000.pt \
    -p prompts/my_music.wav \
    -d 20.0
```

### Example 3: Conservative Generation

More coherent, less random output:

```bash
python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts prompts/my_music.wav \
    --top-k 30 \
    --temperature 0.7
```

### Example 4: Creative Generation

More diverse, experimental output:

```bash
python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts prompts/my_music.wav \
    --top-k -100 \
    --temperature 1.2
```

### Example 5: Batch Processing

Process multiple prompts at once:

```bash
python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts "prompt1.wav|prompt2.wav|prompt3.wav" \
    --output-dir batch_results
```

### Example 6: Custom Model Configuration

If you trained with different settings:

```bash
python3 bin/infer_audiolm.py \
    --checkpoint exp/custom/checkpoint-10000.pt \
    --audio-prompts prompts/test.wav \
    --decoder-dim 512 \
    --nhead 8 \
    --num-decoder-layers 6
```

## 📚 Documentation

### Quick Reference
See `QUICK_REFERENCE_INFERENCE.md` for common commands and parameter tips.

### Complete Guide
See `AUDIO_TO_AUDIO_INFERENCE.md` for:
- Detailed parameter explanations
- Advanced usage examples
- Tips for better generation
- Programmatic API usage
- Comprehensive troubleshooting

### Technical Details
See `INFERENCE_SETUP_COMPLETE.md` for:
- Architecture overview
- Implementation details
- Comparison with original VALLE
- Development notes

## 🎛️ Key Parameters

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `--max-duration` | 10.0 | How many seconds to generate |
| `--top-k` | -100 | Sampling diversity (-100 = unrestricted) |
| `--temperature` | 1.0 | Randomness (lower = safer, higher = creative) |

### Recommended Settings

**For Music**:
```bash
--max-duration 15.0 --top-k -100 --temperature 1.0
```

**For Speech**:
```bash
--max-duration 10.0 --top-k 50 --temperature 0.8
```

**For Sound Effects**:
```bash
--max-duration 5.0 --top-k 30 --temperature 0.7
```

## 🐛 Troubleshooting

### Common Issues

**1. "Checkpoint not found"**
```bash
# Check if file exists
ls exp/VALLE-audio/checkpoint-*.pt

# If missing, you need to train the model first
python3 bin/trainer_audiolm.py [training args]
```

**2. "Module 'torch' not found"**
```bash
# Install PyTorch
pip install torch torchaudio

# Or use conda
conda install pytorch torchaudio -c pytorch
```

**3. "Out of memory"**
```bash
# Reduce generation duration
--max-duration 5.0

# Or use CPU (slower)
CUDA_VISIBLE_DEVICES="" python3 bin/infer_audiolm.py ...
```

**4. "Model configuration mismatch"**
```bash
# Ensure inference config matches training config
# Check your training command for these values:
--decoder-dim 1024 \
--nhead 16 \
--num-decoder-layers 12 \
--num-quantizers 4
```

**5. "Audio prompt too short"**
- Use prompts of at least 3 seconds
- Longer prompts (5+ seconds) give better results

### Getting Help

1. Run verification script: `python3 bin/verify_inference_setup.py`
2. Check complete guide: `AUDIO_TO_AUDIO_INFERENCE.md`
3. See troubleshooting section in the complete guide

## 🔧 Advanced Usage

### Programmatic API

```python
import torch
from valle.models.valle_audio import VALLE_Audio
from valle.data import tokenize_audio, AudioTokenizer32Violin

# Load model
model = VALLE_Audio(d_model=1024, nhead=16, num_layers=12, num_quantizers=4)
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model"])
model.eval()
model.cuda()

# Tokenize prompt
tokenizer = AudioTokenizer32Violin()
audio_tokens = tokenize_audio("Encodec32Violin", "prompt.wav", tokenizer)
audio_tokens = audio_tokens.cuda()
audio_lens = torch.tensor([audio_tokens.shape[1]], device="cuda")

# Generate
with torch.no_grad():
    output = model.inference(
        x=audio_tokens,
        x_lens=audio_lens,
        max_new_tokens=750,
        top_k=-100,
        temperature=1.0,
    )

# Decode and save
audio = tokenizer.decode(output.transpose(1, 2), None)
import torchaudio
torchaudio.save("output.wav", audio[0].cpu(), 32000)
```

## 📊 Performance Notes

- **Speed**: ~1 second of computation per second of audio generated (on GPU)
- **Memory**: ~2-4GB VRAM for default model (1024-dim)
- **Quality**: Best with 5+ second prompts

## 🎯 Next Steps

1. **Train your model** (if not done):
   ```bash
   python3 bin/trainer_audiolm.py --model-name VALLE-audio ...
   ```

2. **Prepare audio prompts**:
   - 3-5 second clips
   - High quality audio
   - Place in `prompts/` directory

3. **Run inference**:
   ```bash
   ./bin/generate_audio.sh -c YOUR_CHECKPOINT -p YOUR_PROMPT
   ```

4. **Experiment**:
   - Try different temperatures
   - Try different prompt lengths
   - Compare different checkpoints

## 📖 Full Documentation Index

| Document | Purpose | Read When |
|----------|---------|-----------|
| `README_INFERENCE.md` | Overview (this file) | Starting out |
| `QUICK_REFERENCE_INFERENCE.md` | Quick commands | Need quick answer |
| `AUDIO_TO_AUDIO_INFERENCE.md` | Complete guide | Learning details |
| `INFERENCE_SETUP_COMPLETE.md` | Technical details | Development work |
| `CHANGES_SUMMARY.md` | What was changed | Understanding changes |

## 🎉 Ready to Go!

Your audio-to-audio inference system is complete and ready to use. Start with:

```bash
cd egs/atepp

# Verify setup
python3 bin/verify_inference_setup.py

# Run inference
./bin/generate_audio.sh \
    -c YOUR_CHECKPOINT.pt \
    -p YOUR_PROMPT.wav

# Check results
ls infer/audio_continuation/
```

Happy generating! 🎵
