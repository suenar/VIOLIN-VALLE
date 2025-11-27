# Audio-to-Audio Inference Setup Complete ✓

## Summary

I've successfully created an inference system for the VALLE-audio model that mimics the structure of the original VALLE model's inference script.

## What Was Created

### 1. Inference Script: `bin/infer_audiolm.py`
A comprehensive Python inference script with:
- Audio prompt loading and tokenization
- Model checkpoint loading with flexible configuration
- Audio generation with customizable parameters
- Saves both full audio (prompt + continuation) and continuation-only
- Support for multiple prompts in batch
- Detailed logging

**Location**: `/workspace/egs/atepp/bin/infer_audiolm.py`

### 2. Wrapper Script: `bin/generate_audio.sh`
A simple bash wrapper for easy command-line usage:
- User-friendly argument parsing
- Built-in help message
- Input validation
- Clear status messages

**Location**: `/workspace/egs/atepp/bin/generate_audio.sh`

### 3. Documentation: `AUDIO_TO_AUDIO_INFERENCE.md`
Comprehensive guide including:
- Quick start examples
- All command-line arguments explained
- Tips for better generation
- Troubleshooting guide
- Programmatic API usage examples
- Comparison with original VALLE

**Location**: `/workspace/AUDIO_TO_AUDIO_INFERENCE.md`

## Model Modifications

### Enhanced `inference()` Function in `valle_audio.py`

Added `return_full_sequence` parameter:
```python
def inference(
    self,
    x: torch.Tensor,
    x_lens: torch.Tensor,
    max_new_tokens: int = 500,
    top_k: int = -100,
    temperature: float = 1.0,
    return_full_sequence: bool = True,  # NEW PARAMETER
) -> torch.Tensor:
```

- When `True`: Returns prompt + generated continuation (full sequence)
- When `False`: Returns only generated continuation

This makes it consistent with typical audio continuation use cases where you want the full output.

## Usage Examples

### Quick Start (using wrapper script)

```bash
cd egs/atepp

# Basic usage
./bin/generate_audio.sh \
    -c exp/VALLE-audio/checkpoint-20000.pt \
    -p prompts/prompt_A.wav

# With custom parameters
./bin/generate_audio.sh \
    -c exp/VALLE-audio/checkpoint-20000.pt \
    -p prompts/prompt_A.wav \
    -d 15.0 -k 50 -t 0.8 \
    -o results -f my_audio
```

### Advanced Usage (direct Python script)

```bash
cd egs/atepp

python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts "./prompts/prompt_A.wav" \
    --output-dir infer/continuation \
    --output-file generated \
    --max-duration 10.0 \
    --top-k -100 \
    --temperature 1.0 \
    --decoder-dim 1024 \
    --nhead 16 \
    --num-decoder-layers 12 \
    --num-quantizers 4
```

### Multiple Prompts

```bash
python3 bin/infer_audiolm.py \
    --checkpoint exp/VALLE-audio/checkpoint-20000.pt \
    --audio-prompts "prompt_A.wav|prompt_B.wav|prompt_C.wav" \
    --output-dir results \
    --output-file batch_gen
```

## Architecture Similarity to Original VALLE

The inference script maintains the same structure as the original VALLE:

| Component | Original VALLE | VALLE-audio |
|-----------|---------------|-------------|
| **Input** | Text/MIDI tokens | Audio tokens |
| **Model Loading** | `get_model()` + checkpoint | `get_model()` + checkpoint |
| **Tokenization** | MIDI/Text tokenizer | Audio tokenizer |
| **Generation** | AR + NAR stages | AR + NAR stages |
| **Output** | Audio waveform | Audio waveform |
| **Script Structure** | `bin/infer.py` | `bin/infer_audiolm.py` |

## Key Features

### 1. Flexible Model Configuration
The script automatically handles:
- Different model sizes (decoder_dim, nhead, num_layers)
- Different quantizer counts
- Different tokenizers (Encodec variants)

### 2. Generation Control
Fine-tune generation with:
- `--max-duration`: Control output length
- `--top-k`: Control diversity (lower = more conservative)
- `--temperature`: Control randomness (lower = more deterministic)

### 3. Dual Output
Automatically saves:
- **Full sequence**: Original prompt + generated continuation
- **Continuation only**: Just the generated part

### 4. Batch Processing
Process multiple prompts in one run for efficiency

## Testing the Setup

### Basic Test

```bash
# Assuming you have a trained checkpoint and a prompt file
cd egs/atepp

# Test with wrapper script
./bin/generate_audio.sh \
    -c exp/VALLE-audio/checkpoint-20000.pt \
    -p prompts/prompt_A.wav \
    -o test_output \
    -d 5.0
```

Expected output:
```
test_output/
├── generated_0_full.wav          # Prompt + 5s continuation
└── generated_0_continuation.wav  # 5s continuation only
```

### Verify Output

```bash
# Check duration
ffprobe test_output/generated_0_full.wav

# Play audio (if you have a player)
ffplay test_output/generated_0_full.wav
```

## Files Modified/Created

### Modified
1. `/workspace/valle/models/valle_audio.py`
   - Enhanced `inference()` method with `return_full_sequence` parameter
   - Line 522-666: Complete inference implementation

### Created
1. `/workspace/egs/atepp/bin/infer_audiolm.py`
   - Main inference script (335 lines)
   
2. `/workspace/egs/atepp/bin/generate_audio.sh`
   - Wrapper script for easy usage (150 lines)
   
3. `/workspace/AUDIO_TO_AUDIO_INFERENCE.md`
   - Complete documentation and user guide

4. `/workspace/INFERENCE_SETUP_COMPLETE.md`
   - This summary file

## Next Steps

1. **Train Your Model** (if not already done):
   ```bash
   python3 bin/trainer_audiolm.py \
       --model-name VALLE-audio \
       --exp-dir exp/VALLE-audio \
       ... [training args]
   ```

2. **Prepare Audio Prompts**:
   - Create 3-5 second audio clips
   - Use clean, high-quality audio
   - Place in `prompts/` directory

3. **Run Inference**:
   ```bash
   ./bin/generate_audio.sh \
       -c exp/VALLE-audio/checkpoint-XXXXX.pt \
       -p prompts/your_prompt.wav
   ```

4. **Experiment with Parameters**:
   - Try different `--temperature` values (0.7 to 1.2)
   - Try different `--top-k` values (30 to 100)
   - Try different `--max-duration` values

## Troubleshooting

If you encounter issues, check:

1. **Model configuration matches training**:
   - decoder_dim, nhead, num_decoder_layers must match training config

2. **Checkpoint exists**:
   - Verify the checkpoint path is correct

3. **Audio prompt is valid**:
   - Must be a valid audio file (wav, mp3, etc.)
   - Should be at least 1 second long

4. **Dependencies installed**:
   - torch, torchaudio, icefall, encodec, etc.

See `AUDIO_TO_AUDIO_INFERENCE.md` for detailed troubleshooting guide.

## Comparison with Original VALLE Inference

| Aspect | Original VALLE | VALLE-audio |
|--------|---------------|-------------|
| **Input** | `--midi-prompts`, `--audio-prompts` | `--audio-prompts` only |
| **Conditioning** | MIDI tokens + optional audio | Audio tokens only |
| **Use Case** | MIDI-to-audio synthesis | Audio continuation |
| **Inference Mode** | `model.inference()` with MIDI | `model.inference()` with audio |
| **Output** | Synthesized audio from MIDI | Continued audio from prompt |
| **Script Name** | `bin/infer.py` | `bin/infer_audiolm.py` |

## Summary

✅ **Inference script created** - Full-featured Python script for audio generation  
✅ **Wrapper script created** - Easy-to-use bash wrapper  
✅ **Documentation complete** - Comprehensive guide with examples  
✅ **Model enhanced** - Added `return_full_sequence` parameter  
✅ **Structure mimics VALLE** - Consistent with original architecture  
✅ **Ready to use** - All scripts executable and documented  

Your audio-to-audio inference system is now complete and ready to use!
