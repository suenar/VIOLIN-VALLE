# Kontakt 8 Quick Start - 5 Minutes ⚡

Get Kontakt 8 working with the script in 5 minutes!

## 🎯 The Problem

```bash
# This WON'T work:
python reaper_midi_to_audio.py -i ./midi -o ./output -p "VST3: Kontakt 8"
# Result: Empty audio files! 🔇
```

**Why?** Kontakt needs an instrument loaded inside it.

## ✅ The Solution

Use **FX Chains** to save your Kontakt configuration with instrument already loaded.

## 🚀 3-Step Setup

### Step 1: Get Help (30 seconds)

```bash
python reaper_midi_to_audio.py --setup-kontakt
```

This shows you the complete setup guide.

### Step 2: Create FX Chain (2 minutes)

**Option A: Automated (Recommended)**
```bash
python create_kontakt_fx_chain.py
```

The script will:
1. Connect to Reaper ✓
2. Load Kontakt ✓
3. Wait for you to select your violin instrument
4. Guide you through saving the FX chain ✓

**Option B: Manual**
1. Open Reaper
2. Add track (Ctrl+T)
3. Add FX: "VST3: Kontakt 8"
4. Load your violin instrument in Kontakt
5. Right-click track → Track FX → Save chain...
6. Name: `Kontakt_Violin.RfxChain`

### Step 3: Use It! (30 seconds)

Find your FX chain file:

**Linux:** `~/.config/REAPER/FXChains/Kontakt_Violin.RfxChain`  
**Mac:** `~/Library/Application Support/REAPER/FXChains/Kontakt_Violin.RfxChain`  
**Windows:** `C:\Users\YourName\AppData\Roaming\REAPER\FXChains\Kontakt_Violin.RfxChain`

Run the script:

```bash
python reaper_midi_to_audio.py \
    -i ./midi_files \
    -o ./output \
    -fx ~/.config/REAPER/FXChains/Kontakt_Violin.RfxChain
```

**Done!** 🎉

## 📋 Example Workflow

```bash
# 1. Setup (one time)
python create_kontakt_fx_chain.py
# → Save as: Kontakt_Violin.RfxChain

# 2. Render MIDI files
python reaper_midi_to_audio.py \
    -i ../samples \
    -o ./violin_output \
    -fx ~/.config/REAPER/FXChains/Kontakt_Violin.RfxChain

# 3. Enjoy your violin audio! 🎻
ls ./violin_output/
```

## 💡 Pro Tips

### Create Multiple Instruments

```bash
# Solo violin
Kontakt_Violin_Solo.RfxChain

# Ensemble violins
Kontakt_Violin_Ensemble.RfxChain

# Other instruments
Kontakt_Cello.RfxChain
Kontakt_Piano.RfxChain
```

Use them:
```bash
python reaper_midi_to_audio.py \
    -i ./midi \
    -o ./output \
    -fx ~/.config/REAPER/FXChains/Kontakt_Violin_Solo.RfxChain
```

### Test Before Batch Processing

```bash
# Test with 1-2 files first
python reaper_midi_to_audio.py \
    -i ./test_midi \
    -o ./test_output \
    -fx /path/to/chain.RfxChain

# Check quality, then batch process everything
```

## 🐛 Troubleshooting

**Can't find FX chain file?**
```bash
# Linux/Mac
find ~ -name "*.RfxChain" 2>/dev/null
```

**Still getting empty audio?**
- Make sure instrument was loaded when you saved the FX chain
- Test the FX chain manually in Reaper first
- Check MIDI file has note data

**Need more help?**
```bash
# Comprehensive guide
cat KONTAKT_GUIDE.md

# Setup instructions
python reaper_midi_to_audio.py --setup-kontakt
```

## 🎓 What's Happening?

1. **FX Chain** = Saved Reaper plugin configuration
2. Includes: Kontakt + loaded instrument + all settings
3. Reusable across all your MIDI files
4. One-time setup, infinite usage! ✨

## 📚 More Info

- **Full guide**: `KONTAKT_GUIDE.md`
- **Main docs**: `README_REAPER.md`
- **Quick start**: `QUICKSTART.md`

---

**Bottom line:** Save Kontakt with instrument loaded → Use the FX chain file → Profit! 🎻
