# Quick Start Guide - Reaper MIDI to Audio

Get started in 5 minutes! 🚀

## Step 1: Prerequisites (2 min)

```bash
# Install reapy library
pip install python-reapy

# Or use requirements file
pip install -r reaper_requirements.txt
```

## Step 2: Configure Reaper (2 min)

1. **Start Reaper**

2. **Enable Web Control:**
   - `Options → Preferences → Control/OSC/web`
   - Click `Add` → Select `Web interface`
   - Click `OK`

3. **Note Your Plugin Name:**
   - Add a track (Ctrl+T)
   - Click `FX` button
   - Find your violin plugin
   - Copy the EXACT name (including "VST3:" prefix, etc.)

## Step 3: Test Your Setup (1 min)

```bash
# Run setup test
python reaper_setup_test.py
```

This will verify:
- ✓ reapy is installed
- ✓ Reaper is running and accessible
- ✓ MIDI import works

## Step 4: Render Your First MIDI File

```bash
# Basic example (using built-in synth)
python reaper_midi_to_audio.py \
    -i ../samples \
    -o ./test_output

# With your violin plugin
python reaper_midi_to_audio.py \
    -i ../samples \
    -o ./test_output \
    -p "VST3: Your Plugin Name (Manufacturer)"
```

## Common Plugin Examples

```bash
# Kontakt 7
-p "VST3: Kontakt 7 (Native Instruments)"

# Spitfire Audio
-p "VST: Solo Violin (Spitfire Audio)"

# Halion
-p "VST3: Halion Sonic SE (Steinberg Media Technologies)"

# Reaper built-in
-p "ReaSynth"
```

## All Command Options

```bash
python reaper_midi_to_audio.py \
    -i INPUT_DIR \          # Directory with MIDI files
    -o OUTPUT_DIR \         # Output directory for WAV files
    -p "PLUGIN_NAME" \      # VST plugin name (optional)
    -sr SAMPLE_RATE         # Sample rate (default: 44100)
```

## Troubleshooting

### Can't Connect to Reaper?
- ✓ Is Reaper running?
- ✓ Web control enabled? (Options → Preferences → Control/OSC/web)
- ✓ Firewall blocking port 8080?

### Plugin Not Loading?
- ✓ Plugin name exactly matches Reaper's FX browser?
- ✓ Plugin scanned? (Options → Preferences → Plug-ins → VST → Re-scan)
- ✓ Include prefix? (`VST3:`, `VST:`, `VSTi:`)

### No Output Files?
- ✓ Check Reaper's render queue
- ✓ Verify output directory is writable
- ✓ MIDI file is valid (test in Reaper manually)

## File Overview

| File | Purpose |
|------|---------|
| `reaper_midi_to_audio.py` | Main script (uses reapy) - **Use this one!** |
| `reaper_midi_to_audio_native.py` | Native ReaScript version (run in Reaper) |
| `reaper_setup_test.py` | Test your setup and find plugins |
| `batch_render_examples.sh` | Example batch processing script |
| `README_REAPER.md` | Comprehensive documentation |
| `QUICKSTART.md` | This file |

## Next Steps

1. ✅ Test with 1-2 MIDI files first
2. ✅ Verify audio quality
3. ✅ Adjust plugin settings if needed
4. ✅ Batch process your full collection

## Getting Help

```bash
# Show all options
python reaper_midi_to_audio.py --help

# Plugin naming help
python reaper_midi_to_audio.py --list-plugins

# Read full documentation
cat README_REAPER.md
```

## Example Workflow

```bash
# 1. Test setup
python reaper_setup_test.py

# 2. Process with your plugin
python reaper_midi_to_audio.py \
    -i ../samples \
    -o ./violin_output \
    -p "VST3: Kontakt 7 (Native Instruments)"

# 3. Check output
ls -lh ./violin_output/
```

---

**Happy rendering! 🎻**

For detailed documentation, see `README_REAPER.md`
