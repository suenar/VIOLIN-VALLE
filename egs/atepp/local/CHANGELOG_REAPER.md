# Changelog - Reaper MIDI to Audio Scripts

## [1.1.0] - 2025-10-28

### Added - Kontakt 8 Support
- **FX Chain support**: New `--fx-chain` option to load pre-configured plugin chains
- **Track Template support**: New `--track-template` option for track templates
- **Kontakt setup guide**: New `--setup-kontakt` flag shows comprehensive setup instructions
- **Interactive FX chain creator**: New `create_kontakt_fx_chain.py` helper script
- **Comprehensive Kontakt guide**: New `KONTAKT_GUIDE.md` with complete workflow documentation

### Features
- Automatic Kontakt detection with helpful warnings
- FX chain file validation
- Support for any multi-instrument VST (not just Kontakt)
- Pre-configured instrument loading via FX chains
- Detailed step-by-step guides for Kontakt setup

### Files Added
- `create_kontakt_fx_chain.py` - Interactive FX chain creation helper
- `KONTAKT_GUIDE.md` - Comprehensive Kontakt integration guide (338 lines)

### Changes
- Enhanced `ReaperMIDIRenderer` class with FX chain and track template support
- Added `_load_fx_chain()` method for loading .RfxChain files
- Updated argument parser with `--fx-chain`, `--track-template`, `--setup-kontakt`
- Improved logging to show FX chain usage
- Better error messages for Kontakt users

---

## [1.0.1] - 2025-10-28

### Fixed
- **CRITICAL FIX**: MIDI items now correctly import at position 0 for each file in batch processing
  - Previously, each subsequent MIDI file would be inserted at the end of the previous one
  - Added `SetEditCurPos(0, True, False)` call before `InsertMedia()` to reset cursor position
  - Fixed in both `reaper_midi_to_audio.py` and `reaper_midi_to_audio_native.py`

### Technical Details
The issue was caused by Reaper's `InsertMedia()` function inserting media at the current edit cursor position. The edit cursor position was accumulating across loop iterations, causing each new MIDI file to be placed after the previous one instead of at time 0.

**Solution**: Explicitly set the edit cursor to position 0 before each MIDI import using:
```python
reapy.reascript_api.SetEditCurPos(0, True, False)  # reapy version
RPR_SetEditCurPos(0, True, False)                   # native version
```

### Files Modified
- `reaper_midi_to_audio.py` (line 129)
- `reaper_midi_to_audio_native.py` (line 75)

---

## [1.0.0] - 2025-10-28

### Added
- Initial release of Reaper MIDI to Audio rendering scripts
- `reaper_midi_to_audio.py` - Main script using reapy library
- `reaper_midi_to_audio_native.py` - Native ReaScript version
- `reaper_setup_test.py` - Setup testing utility
- `batch_render_examples.sh` - Batch processing examples
- Comprehensive documentation (README_REAPER.md, QUICKSTART.md)
- Requirements file (reaper_requirements.txt)

### Features
- Batch MIDI file processing
- VST/VST3i plugin support
- Configurable sample rate
- Progress tracking and logging
- Robust error handling
- Recursive directory scanning
