# ✅ Kontakt 8 - WORKING SOLUTION

## 🎯 The Problem with FX Chains

The FX chain approach has issues loading plugin state programmatically. 

**Better Solution:** Use **Track Templates** instead! ⬇️

---

## ✅ RECOMMENDED SOLUTION: Track Templates

Track templates are more reliable and fully supported by Reaper's API.

### Why Track Templates?
- ✅ Easier to create (built-in Reaper feature)
- ✅ More reliable loading
- ✅ Includes ALL track settings
- ✅ Better API support
- ✅ **Actually works!**

---

## 🚀 Quick Start (3 Steps)

### Step 1: Create Track Template in Reaper

1. **Open Reaper**

2. **Add a track** (Ctrl+T)

3. **Load Kontakt 8**
   - Click FX button on track
   - Add "VST3: Kontakt 8 (Native Instruments)"

4. **Load your violin instrument in Kontakt**
   - Browse to your library
   - Select violin patch
   - Configure settings (velocity, expression, etc.)

5. **Save as Track Template**
   - Right-click the track
   - Select "Save track as template..."
   - Name it: `Kontakt_Violin`
   - Save to default location

6. **Note the file path**
   - Linux: `~/.config/REAPER/TrackTemplates/Kontakt_Violin.RTrackTemplate`
   - Mac: `~/Library/Application Support/REAPER/TrackTemplates/Kontakt_Violin.RTrackTemplate`
   - Windows: `C:\Users\YourName\AppData\Roaming\REAPER\TrackTemplates\Kontakt_Violin.RTrackTemplate`

### Step 2: Use the Template with the Script

**Option A: Use the enhanced v2 script (recommended)**

```bash
python reaper_midi_to_audio_v2.py \
    -i ../samples \
    -o ./violin_output \
    -tt ~/.config/REAPER/TrackTemplates/Kontakt_Violin.RTrackTemplate
```

**Option B: Use the original script**

```bash
python reaper_midi_to_audio.py \
    -i ../samples \
    -o ./violin_output \
    --track-template ~/.config/REAPER/TrackTemplates/Kontakt_Violin.RTrackTemplate
```

### Step 3: Enjoy Your Audio! 🎻

```bash
ls -lh ./violin_output/
```

---

## 📋 Complete Example

```bash
# 1. Create track template in Reaper (manual, one-time)
#    - Add track
#    - Load Kontakt + violin
#    - Save as template: "Kontakt_Violin"

# 2. Process MIDI files
cd /workspace/egs/atepp/local

python reaper_midi_to_audio_v2.py \
    -i ../samples \
    -o ./renders \
    -tt ~/.config/REAPER/TrackTemplates/Kontakt_Violin.RTrackTemplate \
    -sr 48000

# 3. Check output
ls -lh ./renders/
```

---

## 🆚 Comparison: FX Chains vs Track Templates

| Feature | FX Chains | Track Templates |
|---------|-----------|-----------------|
| **Reliability** | ❌ Issues | ✅ Works great |
| **Ease of creation** | Medium | ✅ Easy |
| **API support** | Limited | ✅ Full support |
| **Plugin state** | ❌ Unreliable | ✅ Reliable |
| **Track settings** | No | ✅ Yes |
| **Recommendation** | ❌ Don't use | ✅ **USE THIS** |

---

## 📖 Detailed Instructions

### Creating Track Templates for Different Instruments

#### Solo Violin

```
1. Add track → Load Kontakt → Load solo violin
2. Right-click track → Save track as template
3. Name: Kontakt_Violin_Solo.RTrackTemplate
```

#### Ensemble Violins

```
1. Add track → Load Kontakt → Load ensemble violins
2. Right-click track → Save track as template
3. Name: Kontakt_Violin_Ensemble.RTrackTemplate
```

#### Other Instruments

```
Kontakt_Cello.RTrackTemplate
Kontakt_Viola.RTrackTemplate
Kontakt_Piano.RTrackTemplate
```

### Using Different Templates

```bash
# Violin parts
python reaper_midi_to_audio_v2.py \
    -i ./midi/violin \
    -o ./output/violin \
    -tt ~/.config/REAPER/TrackTemplates/Kontakt_Violin.RTrackTemplate

# Cello parts
python reaper_midi_to_audio_v2.py \
    -i ./midi/cello \
    -o ./output/cello \
    -tt ~/.config/REAPER/TrackTemplates/Kontakt_Cello.RTrackTemplate
```

---

## 🐛 Troubleshooting

### Still Getting Empty Audio?

**Solution 1: Verify Template**
1. In Reaper: Right-click track area → "Insert track from template"
2. Select your template
3. Verify Kontakt loads with instrument
4. If not, recreate the template

**Solution 2: Check MIDI Import**
1. Make sure MIDI file is valid
2. Check that notes are on correct channel
3. Verify Kontakt is listening to the right channel

**Solution 3: Manual Test**
1. Run the script with one file
2. Don't let script finish - pause before render
3. Check Reaper - is instrument loaded?
4. If yes, continue. If no, recreate template.

### Template File Not Found?

```bash
# Find all track templates
find ~ -name "*.RTrackTemplate" 2>/dev/null

# Check default location
ls ~/.config/REAPER/TrackTemplates/  # Linux
ls ~/Library/Application\ Support/REAPER/TrackTemplates/  # Mac
```

### Wrong Instrument Loads?

- Delete the old template in Reaper
- Create new template with correct instrument
- Use new template path in script

---

## 🔄 Migration from FX Chains

If you tried FX chains and they didn't work:

### Old (FX Chain - Don't Use)
```bash
python reaper_midi_to_audio.py \
    -fx /path/to/chain.RfxChain  # ❌ Unreliable
```

### New (Track Template - Use This!)
```bash
python reaper_midi_to_audio_v2.py \
    -tt /path/to/template.RTrackTemplate  # ✅ Reliable
```

---

## ⚡ Quick Reference Card

```bash
# CREATE TEMPLATE (in Reaper, one time):
# 1. Add track
# 2. Add Kontakt + load violin
# 3. Right-click track → Save track as template

# USE TEMPLATE (every time):
python reaper_midi_to_audio_v2.py \
    -i INPUT_DIR \
    -o OUTPUT_DIR \
    -tt /path/to/Kontakt_Violin.RTrackTemplate

# FIND TEMPLATES:
find ~ -name "*.RTrackTemplate" 2>/dev/null
```

---

## 📁 File Locations

### Track Templates Location

| OS | Path |
|---|---|
| Linux | `~/.config/REAPER/TrackTemplates/` |
| Mac | `~/Library/Application Support/REAPER/TrackTemplates/` |
| Windows | `%APPDATA%\REAPER\TrackTemplates\` |

### Scripts to Use

| Script | Status | Use For |
|--------|--------|---------|
| `reaper_midi_to_audio_v2.py` | ✅ Recommended | Track templates |
| `reaper_midi_to_audio.py` | ⚠️ Updated | FX chains (unreliable) |
| `create_kontakt_fx_chain.py` | ❌ Skip | FX chains (don't use) |

---

## ✅ Verification Checklist

Before batch processing:

- [ ] Reaper is running
- [ ] Web control is enabled (Preferences → Control/OSC/web)
- [ ] Track template created with Kontakt + instrument
- [ ] Template file path is correct
- [ ] Tested with 1-2 MIDI files first
- [ ] Audio sounds good
- [ ] Ready for batch processing!

---

## 🎉 Summary

**The Fix:**
1. ❌ ~~FX Chains~~ - Don't use (API limitations)
2. ✅ **Track Templates** - Use this! (Reliable and easy)

**Steps:**
1. Create track template in Reaper (manual, one-time)
2. Use `reaper_midi_to_audio_v2.py` with `-tt` flag
3. Process hundreds of MIDI files automatically!

**Example:**
```bash
python reaper_midi_to_audio_v2.py \
    -i ./midi_files \
    -o ./output \
    -tt ~/.config/REAPER/TrackTemplates/Kontakt_Violin.RTrackTemplate
```

---

## 📞 Additional Help

- **Read:** This file for track template approach
- **Use:** `reaper_midi_to_audio_v2.py` for best results
- **Test:** With 1-2 files before batch processing

**Questions?** Track templates are a standard Reaper feature and much more reliable than FX chains for automation!

---

**Last Updated:** 2025-10-28 (Fixed - using Track Templates)  
**Status:** ✅ Working Solution
