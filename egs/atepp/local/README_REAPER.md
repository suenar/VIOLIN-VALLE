# Reaper MIDI to Audio Rendering Scripts

Scripts to generate violin (or other instrument) audio from MIDI files using Reaper DAW and VST plugins.

## Overview

Two scripts are provided:

1. **`reaper_midi_to_audio.py`** - Uses `reapy` library (recommended for automation)
2. **`reaper_midi_to_audio_native.py`** - Uses native ReaScript API (run from within Reaper)

## Prerequisites

### Required Software

1. **Reaper DAW** - Download from https://www.reaper.fm/
2. **Python 3.7+**
3. **VST/VST3i plugins** - Violin or other instrument plugins installed

### Recommended Violin VST Plugins

- **Kontakt 7** (Native Instruments) - Various violin libraries
- **Spitfire Audio** - BBC Symphony Orchestra, Solo Strings, etc.
- **Vienna Symphonic Library** - Vienna Instruments
- **8Dio** - Various solo and ensemble strings
- **Audio Modeling SWAM** - Physical modeling violin
- **Embertone** - Joshua Bell Violin, Friedlander Violin

## Installation

### Option 1: Using reapy (Recommended)

```bash
# Install reapy library
pip install python-reapy

# Or install from requirements file
pip install -r reaper_requirements.txt
```

### Option 2: Native ReaScript (No installation needed)

The native script runs directly in Reaper's Python environment.

## Reaper Configuration

### 1. Enable ReaScript Python Support

1. Open Reaper
2. Go to: **Options → Preferences → Plug-ins → ReaScript**
3. Check: **"Enable Python for use with ReaScript"**
4. Set Python DLL path if prompted

### 2. Enable Remote Control (for reapy)

1. In Reaper: **Options → Preferences → Control/OSC/web**
2. Click **"Add"**
3. Select **"Web interface"**
4. Set port (default: 8080)
5. Click **"OK"**

### 3. Find Your Plugin Name

To use a specific violin plugin, you need its exact name:

1. Open Reaper
2. Add a track (Ctrl+T)
3. Click the **"FX"** button on the track
4. Browse for your violin plugin
5. **Copy the exact name** shown in the FX browser

Common formats:
- `VST3: Plugin Name (Manufacturer)`
- `VST: Plugin Name (Manufacturer)`  
- `VSTi: Plugin Name`

Examples:
- `VST3: Kontakt 7 (Native Instruments)`
- `VST: Spitfire Audio Solo Violin (Spitfire Audio)`
- `VSTi: Halion Sonic SE`

## Usage

### Script 1: reaper_midi_to_audio.py (with reapy)

This script can be run from the command line while Reaper is running.

#### Basic Usage

```bash
# Process MIDI files with a violin plugin
python reaper_midi_to_audio.py \
    -i ../samples \
    -o ./output_audio \
    -p "VST3: Kontakt 7 (Native Instruments)"

# Process with default MIDI synth (no plugin)
python reaper_midi_to_audio.py \
    -i ../samples \
    -o ./output_audio

# Get help on plugin naming
python reaper_midi_to_audio.py --list-plugins
```

#### Command Line Arguments

- `-i, --input_dir` - Directory containing MIDI files (required)
- `-o, --output_dir` - Directory for output WAV files (required)
- `-p, --plugin` - VST/VST3i plugin name (optional)
- `-sr, --sample_rate` - Sample rate (default: 44100)
- `--list-plugins` - Show plugin naming help

#### Examples

**Example 1: Process samples with Kontakt**
```bash
python reaper_midi_to_audio.py \
    -i ../samples \
    -o ./violin_audio \
    -p "VST3: Kontakt 7 (Native Instruments)" \
    -sr 48000
```

**Example 2: Process prompts folder**
```bash
python reaper_midi_to_audio.py \
    -i ../prompts \
    -o ./rendered_prompts \
    -p "VST: Spitfire Audio Solo Violin (Spitfire Audio)"
```

**Example 3: Batch process all MIDI in data folder**
```bash
python reaper_midi_to_audio.py \
    -i ../data \
    -o ./batch_output \
    -p "VSTi: Halion Sonic SE"
```

### Script 2: reaper_midi_to_audio_native.py (Native ReaScript)

This script runs from within Reaper's ReaScript environment.

#### Setup

1. Open the script in a text editor
2. Edit the configuration section at the top:

```python
INPUT_DIR = "/workspace/egs/atepp/samples"  # Your MIDI directory
OUTPUT_DIR = "/workspace/egs/atepp/output"  # Output directory
PLUGIN_NAME = "VST3: Kontakt 7 (Native Instruments)"  # Your plugin
SAMPLE_RATE = 44100
```

#### Running the Script

**Method 1: Load as ReaScript Action**

1. Open Reaper
2. Go to: **Actions → Show action list**
3. Click **"New action" → "Load ReaScript"**
4. Select `reaper_midi_to_audio_native.py`
5. Click the **"Run"** button

**Method 2: Run from Script Console**

1. Open Reaper
2. Go to: **Actions → Show action list → ReaScript: New...**
3. Copy and paste the script
4. Click **"Run"**

## Workflow

The scripts perform the following steps for each MIDI file:

1. **Clear Project** - Remove existing tracks
2. **Create Track** - Add a new MIDI track
3. **Import MIDI** - Load the MIDI file to the track
4. **Load Plugin** - Apply the specified VST/VST3i instrument
5. **Render** - Export the audio to WAV format
6. **Save** - Write the output file

## Troubleshooting

### Connection Errors

**Error:** "Failed to connect to Reaper"

**Solutions:**
- Ensure Reaper is running
- Enable web control interface in Reaper preferences
- Check firewall settings
- Verify reapy is installed: `pip list | grep reapy`

### Plugin Loading Errors

**Error:** "Failed to load plugin"

**Solutions:**
- Verify plugin name exactly matches what's shown in Reaper's FX browser
- Check plugin is properly installed and scanned by Reaper
- Try running: Reaper → **Options → Preferences → Plug-ins → VST** → **Re-scan**
- Use `--list-plugins` flag to see naming format examples

### Render Issues

**Error:** "Render failed - output file not created"

**Solutions:**
- Check disk space
- Verify output directory is writable
- Ensure MIDI file is valid
- Check Reaper's render queue isn't stuck
- Try rendering manually in Reaper first to test plugin

### Import Errors

**Error:** "Failed to import MIDI file"

**Solutions:**
- Verify MIDI file is valid (open in Reaper manually)
- Check file permissions
- Ensure file path doesn't contain special characters
- Try converting MIDI to Type 1 format

## Advanced Usage

### Custom Render Settings

You can modify the scripts to add custom render settings:

```python
# In reaper_midi_to_audio.py, modify the render settings section:

# Set bit depth (16 or 24)
reapy.reascript_api.GetSetProjectInfo(
    self.project.id, 'RENDER_BITDEPTH', 24, True
)

# Set channels (1=mono, 2=stereo)
reapy.reascript_api.GetSetProjectInfo(
    self.project.id, 'RENDER_CHANNELS', 2, True
)
```

### Batch Processing with Different Plugins

```python
# Process different folders with different plugins
plugins = {
    "../samples/violin": "VST3: Kontakt 7 (Native Instruments)",
    "../samples/viola": "VST3: Spitfire Viola (Spitfire Audio)",
    "../samples/cello": "VST3: Spitfire Cello (Spitfire Audio)"
}

for input_dir, plugin in plugins.items():
    os.system(f'python reaper_midi_to_audio.py -i {input_dir} -o ./output -p "{plugin}"')
```

### Processing with Multiple Plugin Chains

Modify the script to add multiple effects:

```python
# Add reverb after instrument
fx1 = track.add_fx(plugin_name)  # Instrument
fx2 = track.add_fx("VST: ReaVerb (Cockos)")  # Reverb
```

## Performance Tips

1. **Close unnecessary programs** - DAW rendering is CPU-intensive
2. **Increase buffer size** - Reaper → Options → Preferences → Audio → Device → Request block size
3. **Disable real-time effects** - Only render necessary FX
4. **Use SSD storage** - Faster read/write for audio files
5. **Batch process overnight** - For large MIDI collections

## Example: Complete Workflow

```bash
# 1. Install dependencies
pip install python-reapy

# 2. Start Reaper and enable web control

# 3. Check plugin name
python reaper_midi_to_audio.py --list-plugins

# 4. Process your MIDI files
python reaper_midi_to_audio.py \
    -i /workspace/egs/atepp/samples \
    -o /workspace/egs/atepp/violin_renders \
    -p "VST3: Kontakt 7 (Native Instruments)" \
    -sr 44100

# 5. Verify output
ls -lh /workspace/egs/atepp/violin_renders/
```

## API References

- **Reaper ReaScript API**: https://www.reaper.fm/sdk/reascript/reascripthelp.html
- **reapy Documentation**: https://github.com/RomeoDespres/reapy
- **Reaper Forum**: https://forum.cockos.com/forumdisplay.php?f=20

## License

These scripts are provided as-is for use with Reaper DAW. Follow your plugin licenses for commercial use.

## Support

For issues:
1. Check Reaper is running and configured correctly
2. Verify plugin names match exactly
3. Test manual rendering in Reaper first
4. Check Reaper's console for error messages (View → Show REAPER console)

---

**Note**: These scripts automate Reaper's rendering process. Always test with a few files first before batch processing large collections!
