# 🎻 START HERE - Kontakt 8 + Reaper Script

**Your Question:** How to automatically load instruments inside Kontakt 8 for batch MIDI rendering?

**The Answer:** Use FX Chains! Follow the 3 steps below. ⬇️

---

## ⚡ Quick Start (5 Minutes)

### 1️⃣ Install Requirements

```bash
pip install python-reapy
```

### 2️⃣ Create FX Chain with Kontakt + Violin

```bash
cd /workspace/egs/atepp/local
python create_kontakt_fx_chain.py
```

**This will:**
- Connect to Reaper ✓
- Load Kontakt 8 ✓
- **Wait for you to load your violin instrument** ← Manual step!
- Guide you through saving the FX chain ✓
- Show you the file path ✓

### 3️⃣ Process Your MIDI Files

```bash
python reaper_midi_to_audio.py \
    -i ../samples \
    -o ./violin_output \
    -fx ~/.config/REAPER/FXChains/Kontakt_Violin.RfxChain
```

**Done!** 🎉

---

## 📖 What Just Happened?

1. **FX Chain** = Saved Reaper configuration with Kontakt + instrument loaded
2. Script loads this FX chain automatically for each MIDI file
3. No more manual clicking - fully automated! ✅

---

## 🎯 Complete Example

```bash
# 1. One-time setup
python create_kontakt_fx_chain.py
# → You load violin in Kontakt
# → Save as: Kontakt_Violin.RfxChain

# 2. Process files (repeat as needed)
python reaper_midi_to_audio.py \
    -i ../samples \
    -o ./renders \
    -fx ~/.config/REAPER/FXChains/Kontakt_Violin.RfxChain \
    -sr 48000

# 3. Check output
ls -lh ./renders/
```

---

## 📋 Available Commands

```bash
# Create FX chain interactively
python create_kontakt_fx_chain.py

# Show Kontakt setup guide
python reaper_midi_to_audio.py --setup-kontakt

# Test your Reaper connection
python reaper_setup_test.py

# Show all script options
python reaper_midi_to_audio.py --help
```

---

## 📚 Documentation

| Read This | When |
|-----------|------|
| **SOLUTION_SUMMARY.md** | Complete overview of the solution |
| **KONTAKT_QUICKSTART.md** | 5-minute quick start guide |
| **KONTAKT_GUIDE.md** | Deep dive - everything about Kontakt integration |
| **README_REAPER.md** | General script documentation |
| **QUICKSTART.md** | Basic script quick start |

---

## 🔍 Where Are My FX Chains Saved?

- **Linux:** `~/.config/REAPER/FXChains/`
- **Mac:** `~/Library/Application Support/REAPER/FXChains/`
- **Windows:** `C:\Users\YourName\AppData\Roaming\REAPER\FXChains\`

```bash
# Find them
find ~ -name "*.RfxChain" 2>/dev/null
```

---

## 💡 Pro Tips

### Create Multiple Instruments

```bash
# Run the helper multiple times for different instruments
python create_kontakt_fx_chain.py  # → Kontakt_Violin.RfxChain
python create_kontakt_fx_chain.py  # → Kontakt_Cello.RfxChain
python create_kontakt_fx_chain.py  # → Kontakt_Piano.RfxChain
```

### Different Articulations

```bash
Kontakt_Violin_Legato.RfxChain
Kontakt_Violin_Staccato.RfxChain
Kontakt_Violin_Pizzicato.RfxChain
```

### Batch Different Sections

```bash
python reaper_midi_to_audio.py -i ./violin_midi -o ./out/violin \
    -fx ~/.config/REAPER/FXChains/Kontakt_Violin.RfxChain

python reaper_midi_to_audio.py -i ./cello_midi -o ./out/cello \
    -fx ~/.config/REAPER/FXChains/Kontakt_Cello.RfxChain
```

---

## 🐛 Troubleshooting

### Empty Audio?

1. Test FX chain manually in Reaper
2. Make sure instrument was loaded when you saved it
3. Recreate the FX chain

### Can't Find FX Chain File?

```bash
# Search for it
find ~ -name "*.RfxChain" 2>/dev/null

# Or use full path
python reaper_midi_to_audio.py ... \
    -fx /full/absolute/path/to/Kontakt_Violin.RfxChain
```

### Connection Error?

1. Make sure Reaper is running
2. Enable web control: Options → Preferences → Control/OSC/web
3. Run: `python reaper_setup_test.py`

---

## ✅ What Was Done to Fix Your Issue

### The Problem
Simply loading Kontakt produces empty audio because no instrument is selected.

### The Solution
Added **FX Chain support** to the script:
- New `--fx-chain` option
- New `create_kontakt_fx_chain.py` helper
- Comprehensive documentation
- Interactive setup guide

### The Result
You can now batch process hundreds of MIDI files with Kontakt automatically loading your violin instrument. No manual clicking required! 🎻✨

---

## 🎓 Understanding the Workflow

```
┌─────────────┐
│ MIDI Files  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────────┐
│   Reaper    │ ◄───│ FX Chain File    │
└──────┬──────┘     │ (Kontakt+Violin) │
       │            └──────────────────┘
       │ Loads instrument automatically
       │
       ▼
┌─────────────┐
│  Audio WAV  │
└─────────────┘
```

---

## 🚀 Next Steps

1. **Try it now:**
   ```bash
   python create_kontakt_fx_chain.py
   ```

2. **Process some MIDI files:**
   ```bash
   python reaper_midi_to_audio.py -i INPUT -o OUTPUT -fx CHAIN_FILE
   ```

3. **Read more details:**
   ```bash
   cat SOLUTION_SUMMARY.md
   ```

---

## 📞 Need More Help?

```bash
# Interactive Kontakt guide
python reaper_midi_to_audio.py --setup-kontakt

# Complete documentation
cat KONTAKT_GUIDE.md

# All script options
python reaper_midi_to_audio.py --help
```

---

**Bottom Line:** Create an FX chain once with Kontakt + your instrument loaded, then use it forever! 🎉

**Key File:** `SOLUTION_SUMMARY.md` - Read this for complete details!
