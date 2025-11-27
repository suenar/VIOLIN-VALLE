# VALLE-Audio Inference - Quick Reference Card

## 🚀 Quick Start

```bash
cd egs/atepp

# Generate audio continuation
./bin/generate_audio.sh \
    -c exp/VALLE-audio/checkpoint-20000.pt \
    -p prompts/prompt_A.wav
```

## 📋 Common Commands

### Basic Generation
```bash
python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts prompts/prompt_A.wav \
    --output-dir results
```

### Conservative Generation (coherent, less random)
```bash
python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts prompts/prompt_A.wav \
    --top-k 30 --temperature 0.7
```

### Creative Generation (diverse, more random)
```bash
python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts prompts/prompt_A.wav \
    --top-k -100 --temperature 1.2
```

### Long Generation (15 seconds)
```bash
python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts prompts/prompt_A.wav \
    --max-duration 15.0
```

### Batch Generation (multiple prompts)
```bash
python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts "prompt_A.wav|prompt_B.wav|prompt_C.wav"
```

## 🎛️ Key Parameters

| Parameter | Default | Description | Tips |
|-----------|---------|-------------|------|
| `--max-duration` | 10.0 | Max seconds to generate | Longer = more memory |
| `--top-k` | -100 | Top-k sampling | 30-50 = conservative, -100 = diverse |
| `--temperature` | 1.0 | Sampling temperature | 0.7 = safe, 1.2 = creative |

## 📁 Output Files

Each generation creates 2 files:
- `{name}_full.wav` - Prompt + generated continuation
- `{name}_continuation.wav` - Generated part only

## 🔧 Model Config (must match training)

```bash
--decoder-dim 1024 \
--nhead 16 \
--num-decoder-layers 12 \
--num-quantizers 4
```

## 📊 Typical Generation Settings

| Use Case | top-k | temperature | max-duration |
|----------|-------|-------------|--------------|
| **Music continuation** | -100 | 1.0 | 10-15s |
| **Speech continuation** | 50 | 0.8 | 5-10s |
| **Sound effects** | 30 | 0.7 | 3-5s |
| **Experimental** | -100 | 1.2-1.5 | 15-20s |

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--max-duration` |
| Too repetitive | Increase `--temperature` or use `--top-k -100` |
| Too noisy | Decrease `--temperature` or use `--top-k 30-50` |
| Checkpoint error | Verify model config matches training |

## 📖 Full Documentation

See `AUDIO_TO_AUDIO_INFERENCE.md` for complete guide.

## 🔗 Files

- **Main script**: `egs/atepp/bin/infer_audiolm.py`
- **Wrapper**: `egs/atepp/bin/generate_audio.sh`
- **Model**: `valle/models/valle_audio.py`
- **Docs**: `AUDIO_TO_AUDIO_INFERENCE.md`
