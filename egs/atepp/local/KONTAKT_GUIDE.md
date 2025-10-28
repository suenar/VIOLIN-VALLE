# Kontakt 8 Integration Guide

Complete guide for using Kontakt 8 (or any multi-instrument VST) with the Reaper MIDI rendering scripts.

## 🎯 The Problem

When you load Kontakt 8 directly using:
```bash
python reaper_midi_to_audio.py -i ./midi -o ./output -p "VST3: Kontakt 8"
```

**Result:** Empty audio files! 🔇

**Why?** Kontakt is a plugin **host** - it needs you to load a specific instrument inside it. Simply loading the plugin doesn't load any sounds.

## ✅ The Solution: FX Chains

Use Reaper's **FX Chain** feature to save your Kontakt configuration (with instrument loaded) and reuse it automatically.

## 📋 Quick Start (3 Steps)

### Step 1: Create the FX Chain

**Option A: Use the Helper Script (Easiest)**
```bash
python create_kontakt_fx_chain.py
```

The script will:
- Connect to Reaper
- Load Kontakt
- Guide you through instrument selection
- Help you save the FX chain

**Option B: Manual Setup**
1. Open Reaper
2. Add a track (Ctrl+T)
3. Add FX: "VST3: Kontakt 8 (Native Instruments)"
4. In Kontakt, load your violin instrument
5. Right-click the track → Track FX → Save chain...
6. Name it: `Kontakt_Violin.RfxChain`
7. Save to FXChains folder

### Step 2: Find Your FX Chain File

FX chains are saved to:

| OS | Default Location |
|---|---|
| **Windows** | `C:\Users\YourName\AppData\Roaming\REAPER\FXChains\` |
| **Mac** | `~/Library/Application Support/REAPER/FXChains/` |
| **Linux** | `~/.config/REAPER/FXChains/` |

### Step 3: Use It!

```bash
python reaper_midi_to_audio.py \
    -i ./midi_files \
    -o ./output \
    -fx "/path/to/REAPER/FXChains/Kontakt_Violin.RfxChain"
```

## 🎻 Detailed Walkthrough

### Creating Multiple Instrument Presets

You can create different FX chains for different instruments:

```bash
# Solo violin (expressive, vibrato)
Kontakt_Violin_Solo.RfxChain

# Ensemble violins (section sound)
Kontakt_Violin_Ensemble.RfxChain

# Staccato violin (short notes)
Kontakt_Violin_Staccato.RfxChain

# Other instruments
Kontakt_Cello.RfxChain
Kontakt_Viola.RfxChain
Kontakt_Piano.RfxChain
```

### Configuring Kontakt Settings

When setting up your instrument in Kontakt, consider configuring:

1. **Instrument/Patch Selection**
   - Browse to your library (e.g., "Solo Violin")
   - Select specific articulation (legato, staccato, etc.)

2. **Performance Settings**
   - Velocity range/response
   - Expression mapping
   - Vibrato intensity
   - Legato transitions

3. **Output Settings**
   - Output channel (usually 1-2 for stereo)
   - Volume level
   - Pan position

4. **Effects**
   - Built-in reverb (or disable if adding separate reverb)
   - EQ adjustments
   - Any other built-in processing

All these settings will be saved in the FX chain!

## 💡 Advanced Usage

### Example: Full Workflow

```bash
# 1. Create FX chain with helper
python create_kontakt_fx_chain.py
# → Saves: ~/.config/REAPER/FXChains/Kontakt_Violin.RfxChain

# 2. Process MIDI files
python reaper_midi_to_audio.py \
    -i ../samples \
    -o ./violin_renders \
    -fx ~/.config/REAPER/FXChains/Kontakt_Violin.RfxChain \
    -sr 48000

# 3. Check output
ls -lh ./violin_renders/
```

### Batch Processing Different Instruments

```bash
# Process different MIDI sets with different instruments

# Violin parts
python reaper_midi_to_audio.py \
    -i ./midi/violin \
    -o ./output/violin \
    -fx ~/FXChains/Kontakt_Violin.RfxChain

# Cello parts
python reaper_midi_to_audio.py \
    -i ./midi/cello \
    -o ./output/cello \
    -fx ~/FXChains/Kontakt_Cello.RfxChain

# Piano parts
python reaper_midi_to_audio.py \
    -i ./midi/piano \
    -o ./output/piano \
    -fx ~/FXChains/Kontakt_Piano.RfxChain
```

### Using Different Libraries

Kontakt works with many third-party libraries. The same FX chain approach works for:

- **Native Instruments Libraries**
  - Kontakt Factory Library
  - Session Strings
  - Symphonic Series
  
- **Spitfire Audio** (when loaded in Kontakt)
  - Any Kontakt-compatible Spitfire libraries
  
- **8Dio**
  - Adagio Violins
  - Century Strings
  - Solo Violin
  
- **Embertone**
  - Joshua Bell Violin
  - Friedlander Violin
  
- **Vienna Symphonic Library**
  - Any Kontakt-compatible Vienna instruments

## 🔧 Troubleshooting

### Issue: FX Chain Not Loading

**Symptoms:** Error message about FX chain file not found

**Solutions:**
1. Check the file path is correct
2. Use absolute paths, not relative
3. Make sure .RfxChain extension is included
4. Verify file exists: `ls "/path/to/chain.RfxChain"`

### Issue: Silent Audio Output

**Symptoms:** Files are created but contain silence

**Solutions:**
1. Open the FX chain file in Reaper manually to verify
2. Check Kontakt has instrument loaded when you saved the chain
3. Verify MIDI file has note data
4. Check Kontakt's output routing (should go to track)
5. Ensure Kontakt's instrument is on the correct MIDI channel

### Issue: Wrong Instrument Loads

**Symptoms:** Different instrument than expected

**Solutions:**
1. Create a new FX chain with the correct instrument
2. Make sure you saved the FX chain AFTER loading the instrument
3. Test the FX chain manually in Reaper first

### Issue: FX Chain File Location Unknown

**Solutions:**
```bash
# Linux/Mac
find ~ -name "*.RfxChain" 2>/dev/null

# List all FX chains in default location
ls ~/.config/REAPER/FXChains/  # Linux
ls ~/Library/Application\ Support/REAPER/FXChains/  # Mac
dir %APPDATA%\REAPER\FXChains\  # Windows
```

## 📊 Comparison: Methods

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Direct Plugin** | Simple | Requires manual selection | Single-use, testing |
| **FX Chain** | Reusable, automated | One-time setup | Batch processing ✅ |
| **Track Template** | Full track config | Larger files | Complex setups |

## 🎓 Understanding FX Chains

### What is an FX Chain?

An FX chain is a Reaper file (`.RfxChain`) that stores:
- Plugin name and type
- Plugin state (including loaded instruments)
- Parameter values
- Routing configuration

### What Gets Saved?

✅ **Saved:**
- Kontakt plugin settings
- Loaded instrument/patch
- All parameter values
- Effect settings within Kontakt
- Output routing

❌ **Not Saved:**
- Track volume/pan (use Track Templates for this)
- Track routing/sends
- Track automation
- MIDI data

## 📁 File Structure

After setup, your directory structure might look like:

```
~/.config/REAPER/FXChains/
├── Kontakt_Violin_Solo.RfxChain
├── Kontakt_Violin_Ensemble.RfxChain
├── Kontakt_Violin_Staccato.RfxChain
├── Kontakt_Cello.RfxChain
├── Kontakt_Viola.RfxChain
└── Kontakt_Piano.RfxChain
```

Use them in your scripts:
```bash
python reaper_midi_to_audio.py \
    -i ./input \
    -o ./output \
    -fx ~/.config/REAPER/FXChains/Kontakt_Violin_Solo.RfxChain
```

## 🚀 Pro Tips

1. **Name your chains descriptively**
   - Good: `Kontakt_Violin_Solo_Legato_Vibrato.RfxChain`
   - Bad: `chain1.RfxChain`

2. **Test before batch processing**
   - Process 1-2 MIDI files first
   - Verify audio quality
   - Then process full batch

3. **Version your chains**
   - `Kontakt_Violin_v1.RfxChain`
   - `Kontakt_Violin_v2_more_reverb.RfxChain`
   - Keep track of what works best

4. **Document your settings**
   - Create a text file documenting each FX chain
   - Note which Kontakt library/patch was used
   - List any special configuration

5. **Organize by project**
   ```
   FXChains/
   ├── project_a/
   │   ├── violin.RfxChain
   │   └── cello.RfxChain
   └── project_b/
       └── strings_ensemble.RfxChain
   ```

## 📚 Additional Resources

- **Reaper Manual**: https://www.reaper.fm/guides.php
- **Kontakt Manual**: Native Instruments website
- **FX Chains Tutorial**: Search "Reaper FX chains" on YouTube
- **This Script's Docs**: See `README_REAPER.md`

## ⚡ Quick Reference

```bash
# Show Kontakt setup guide
python reaper_midi_to_audio.py --setup-kontakt

# Create FX chain interactively
python create_kontakt_fx_chain.py

# Use FX chain for rendering
python reaper_midi_to_audio.py \
    -i INPUT_DIR \
    -o OUTPUT_DIR \
    -fx /path/to/chain.RfxChain

# Test your setup
python reaper_setup_test.py
```

---

**Remember:** The key insight is that Kontakt needs an instrument loaded. FX chains save that entire configuration so it's automatically loaded every time! 🎻✨
