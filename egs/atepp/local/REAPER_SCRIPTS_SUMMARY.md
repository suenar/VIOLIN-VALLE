# Reaper MIDI to Audio Scripts - Summary

Created: 2025-10-28

## 📦 Files Created

All files are located in `/workspace/egs/atepp/local/`

### Main Scripts

1. **reaper_midi_to_audio.py** ⭐ (Recommended)
   - Full-featured script using reapy library
   - Run from command line while Reaper is running
   - Best for automation and batch processing
   - ~400 lines of well-documented code

2. **reaper_midi_to_audio_native.py**
   - Uses native ReaScript API (no external dependencies)
   - Run from within Reaper's ReaScript environment
   - Good for integration into Reaper workflows
   - ~300 lines of code

3. **reaper_setup_test.py**
   - Tests your Reaper setup
   - Verifies connection and capabilities
   - Helps discover available plugins
   - Run before first use

### Documentation

4. **README_REAPER.md**
   - Comprehensive 400+ line documentation
   - Installation instructions
   - Detailed usage examples
   - Troubleshooting guide
   - API references

5. **QUICKSTART.md**
   - 5-minute quick start guide
   - Step-by-step setup
   - Common examples
   - Quick troubleshooting

6. **reaper_requirements.txt**
   - Python dependencies
   - Install with: `pip install -r reaper_requirements.txt`

### Utilities

7. **batch_render_examples.sh**
   - Example batch processing script
   - Shows multiple rendering scenarios
   - Easy to customize

## 🎯 Key Features

### Script Capabilities

✅ **Load MIDI files** from any directory (recursive search)
✅ **Apply VST/VST3i plugins** (Kontakt, Spitfire, etc.)
✅ **Batch processing** - handle multiple files automatically
✅ **Configurable rendering** - sample rate, format, etc.
✅ **Error handling** - graceful failures with helpful messages
✅ **Progress tracking** - see what's being processed
✅ **Plugin discovery** - help finding plugin names

### Supported Workflows

- Single MIDI file rendering
- Batch directory processing
- Recursive subdirectory processing
- Custom output directory structures
- Multiple plugin configurations
- Quality presets (44.1kHz, 48kHz, etc.)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /workspace/egs/atepp/local
pip install python-reapy
```

### 2. Configure Reaper

- Start Reaper
- Enable web control: `Options → Preferences → Control/OSC/web → Add → Web interface`

### 3. Find Your Plugin

- Add track (Ctrl+T)
- Click FX button
- Note exact plugin name (e.g., "VST3: Kontakt 7 (Native Instruments)")

### 4. Run the Script

```bash
# Test setup
python reaper_setup_test.py

# Render MIDI files
python reaper_midi_to_audio.py \
    -i ../samples \
    -o ./output \
    -p "VST3: Your Plugin Name"
```

## 📋 Usage Examples

### Basic Examples

```bash
# With plugin
python reaper_midi_to_audio.py \
    -i /path/to/midi \
    -o /path/to/output \
    -p "VST3: Kontakt 7 (Native Instruments)"

# Without plugin (default MIDI synth)
python reaper_midi_to_audio.py \
    -i /path/to/midi \
    -o /path/to/output

# High quality (48kHz)
python reaper_midi_to_audio.py \
    -i /path/to/midi \
    -o /path/to/output \
    -p "VST3: Your Plugin" \
    -sr 48000
```

### Project-Specific Examples

```bash
# Render samples folder
python reaper_midi_to_audio.py \
    -i ../samples \
    -o ./rendered_samples \
    -p "VST3: Kontakt 7 (Native Instruments)"

# Render prompts folder
python reaper_midi_to_audio.py \
    -i ../prompts \
    -o ./rendered_prompts \
    -p "VST: Spitfire Audio Solo Violin"
```

## 🎻 Recommended Violin Plugins

The scripts work with any VST/VST3i plugin. Popular options:

- **Kontakt 7** (Native Instruments) - Supports many violin libraries
- **Spitfire Audio** - BBC SO, Solo Strings, etc.
- **Vienna Instruments** - Vienna Symphonic Library
- **8Dio** - Various string libraries
- **Audio Modeling SWAM** - Physical modeling
- **Embertone** - Joshua Bell Violin, Friedlander Violin

## 🔧 Command Line Arguments

| Argument | Short | Description | Required | Default |
|----------|-------|-------------|----------|---------|
| `--input_dir` | `-i` | MIDI input directory | Yes | - |
| `--output_dir` | `-o` | WAV output directory | Yes | - |
| `--plugin` | `-p` | VST plugin name | No | None |
| `--sample_rate` | `-sr` | Sample rate (Hz) | No | 44100 |
| `--list-plugins` | - | Show plugin naming help | No | - |

## 📖 Documentation

- **Quick Start**: Read `QUICKSTART.md` (5-minute setup)
- **Full Docs**: Read `README_REAPER.md` (comprehensive guide)
- **Test Setup**: Run `python reaper_setup_test.py`
- **Help**: Run `python reaper_midi_to_audio.py --help`

## 🐛 Troubleshooting

### Common Issues

**Can't connect to Reaper**
→ Make sure Reaper is running and web control is enabled

**Plugin not loading**
→ Check exact plugin name in Reaper's FX browser

**No output files**
→ Check Reaper's render queue and output directory permissions

**Import errors**
→ Install reapy: `pip install python-reapy`

See `README_REAPER.md` for detailed troubleshooting.

## 🔗 Resources

- **Reaper API**: https://www.reaper.fm/sdk/reascript/reascripthelp.html
- **reapy GitHub**: https://github.com/RomeoDespres/reapy
- **Reaper Forum**: https://forum.cockos.com/forumdisplay.php?f=20

## 📁 File Structure

```
/workspace/egs/atepp/local/
├── reaper_midi_to_audio.py          # Main script ⭐
├── reaper_midi_to_audio_native.py   # Native ReaScript version
├── reaper_setup_test.py             # Setup test utility
├── batch_render_examples.sh         # Batch processing examples
├── reaper_requirements.txt          # Python dependencies
├── README_REAPER.md                 # Full documentation
├── QUICKSTART.md                    # Quick start guide
└── REAPER_SCRIPTS_SUMMARY.md        # This file
```

## ✅ Script Features

### Implemented Features

- ✅ MIDI file discovery (recursive)
- ✅ VST/VST3i plugin loading
- ✅ Batch processing
- ✅ Progress tracking
- ✅ Error handling
- ✅ Logging
- ✅ Sample rate configuration
- ✅ Output directory creation
- ✅ File naming preservation
- ✅ Success/failure reporting

### Design Highlights

- **Robust error handling** - Graceful failures with helpful messages
- **Progress logging** - See what's happening in real-time
- **Flexible configuration** - Command-line arguments for all options
- **Well documented** - Comprehensive inline comments
- **Type hints** - Clear function signatures
- **Modular design** - Easy to extend and customize

## 🎓 Learning Resources

The scripts demonstrate:
- Reaper API usage (reapy and native)
- Audio rendering automation
- VST plugin management
- Batch file processing
- Command-line argument parsing
- Python logging best practices
- Error handling patterns

## 🔄 Workflow

```
MIDI Files → Reaper → VST Plugin → Render → WAV Files
     ↓          ↓          ↓           ↓         ↓
   Load     Import     Apply      Process    Save
```

The scripts automate this entire pipeline!

## 💡 Tips

1. **Test first** - Always test with 1-2 files before batch processing
2. **Plugin names** - Copy exact names from Reaper's FX browser
3. **Performance** - Close other apps for faster rendering
4. **Quality** - Use 48kHz for professional projects
5. **Backups** - Keep original MIDI files safe

## 🎉 Ready to Use!

Everything is set up and ready to go. Just:

1. Install reapy: `pip install python-reapy`
2. Start Reaper and enable web control
3. Run: `python reaper_midi_to_audio.py -i INPUT -o OUTPUT -p "PLUGIN"`

**Happy rendering! 🎻**

---

*For questions or issues, refer to README_REAPER.md or the Reaper forum.*
