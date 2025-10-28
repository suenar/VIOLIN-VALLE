# Changelog - Reaper MIDI to Audio Scripts

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
