# Changes to prepare_gigamidi Script

## Summary
Modified the script to ensure proper filtering of recordings and supervisions that lack valid transcriptions.

## Key Improvements

### 1. **Enhanced Transcript Validation**
```python
# OLD: Just store paths without validation
transcript_dict[file.stem] = str(file)

# NEW: Validate files exist before adding
for file in transcript_files:
    if not file.is_file():
        logging.warning(f"Transcript file does not exist: {file}")
        continue
    transcript_dict[file.stem] = str(file)
```

### 2. **Multiple Validation Checks**
Added comprehensive checks for each audio file:
- ✅ Check if transcript exists in dictionary
- ✅ Verify transcript file path is valid and exists
- ✅ Verify audio file exists
- ✅ Validate recording can be loaded
- ✅ Check recording duration > 0

### 3. **Better Error Tracking**
```python
skipped_no_transcript = []     # Track files without transcripts
skipped_invalid_audio = []     # Track invalid audio files
skipped_zero_duration = []     # Track zero-duration files
```

### 4. **Detailed Statistics Logging**
After processing each dataset part:
```
=== TRAIN Statistics ===
Valid recordings: 15234
Valid supervisions: 15234
Skipped (no transcript): 127
Skipped (invalid audio): 43
Skipped (zero duration): 12
```

### 5. **Robust Filename Parsing**
```python
# OLD: Could fail if filename format is unexpected
speaker = audio_path.stem.split("_")[0]
language = audio_path.stem.split("_")[1]

# NEW: Validate before splitting
parts = audio_path.stem.split("_")
if len(parts) < 2:
    logging.warning(f"Invalid filename format: {audio_path}")
    continue
speaker = parts[0]
language = parts[1]
```

### 6. **Double-Check Transcript File**
```python
# Check 1: In dictionary?
if idx not in transcript_dict:
    skipped_no_transcript.append(idx)
    continue

# Check 2: File actually exists?
transcript_path = Path(transcript_dict[idx])
if not transcript_path.is_file():
    logging.warning(f"Transcript file missing: {transcript_path}")
    skipped_no_transcript.append(idx)
    del transcript_dict[idx]  # Clean up
    continue
```

### 7. **Safe Recording Creation**
```python
try:
    recording = Recording.from_file(audio_path)
except Exception as e:
    logging.error(f"Failed to load recording {audio_path}: {e}")
    skipped_invalid_audio.append(str(audio_path))
    continue
```

### 8. **Assertions to Ensure Consistency**
```python
# Ensure recordings and supervisions are always in sync
assert len(recordings) == len(supervisions), \
    f"Mismatch: {len(recordings)} recordings vs {len(supervisions)} supervisions"
```

### 9. **Better Logging Configuration**
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### 10. **Added Tokenizer Argument**
```python
parser.add_argument(
    "--tokenizer",
    type=str,
    default="expression",
    choices=["expression", "remi"],
    help="Tokenizer type to use.",
)
```

## Usage

### Basic Usage
```bash
python prepare_gigamidi_filtered.py \
    --corpus-dir /path/to/gigamidi \
    --output-dir /path/to/output
```

### With Tokenizer Selection
```bash
python prepare_gigamidi_filtered.py \
    --corpus-dir /path/to/gigamidi \
    --output-dir /path/to/output \
    --tokenizer expression
```

## What Gets Filtered Out

The script will **exclude** recordings and supervisions if:

1. ❌ No transcript file exists for the audio
2. ❌ Transcript file path is invalid
3. ❌ Audio file doesn't exist
4. ❌ Audio file can't be loaded
5. ❌ Audio duration is 0 or negative
6. ❌ Filename format is invalid (missing speaker/language)

## Output

### Console Output Example
```
2024-01-15 10:30:45 - INFO - Found 15456 valid transcripts
2024-01-15 10:30:46 - INFO - Processing gigamidi subset: train
2024-01-15 10:30:47 - INFO - Found 15500 audio files in train
2024-01-15 10:35:22 - INFO - === TRAIN Statistics ===
2024-01-15 10:35:22 - INFO - Valid recordings: 15234
2024-01-15 10:35:22 - INFO - Valid supervisions: 15234
2024-01-15 10:35:22 - INFO - Skipped (no transcript): 127
2024-01-15 10:35:22 - INFO - Skipped (invalid audio): 43
2024-01-15 10:35:22 - INFO - Skipped (zero duration): 12
2024-01-15 10:35:23 - INFO - Final train: 15234 recordings, 15234 supervisions
2024-01-15 10:35:24 - INFO - Saved manifests to /path/to/output
```

### Files Generated
- `gigamidi_recordings_train.jsonl.gz`
- `gigamidi_supervisions_train.jsonl.gz`
- `gigamidi_recordings_validation.jsonl.gz`
- `gigamidi_supervisions_validation.jsonl.gz`
- `gigamidi_recordings_test.jsonl.gz`
- `gigamidi_supervisions_test.jsonl.gz`

## Benefits

1. **Data Integrity**: Ensures recordings and supervisions are always paired
2. **Better Debugging**: Detailed logs show exactly what was filtered and why
3. **Robustness**: Handles edge cases (missing files, corrupt audio, etc.)
4. **Statistics**: Clear visibility into data quality issues
5. **No Silent Failures**: Every skip is logged with reason

## Comparison: Old vs New

| Feature | Old Script | New Script |
|---------|-----------|------------|
| Transcript validation | Basic check | File existence verified |
| Error tracking | Minimal logging | Detailed categorization |
| Statistics | None | Per-category counts |
| Exception handling | None | Try-catch for recording loading |
| Filename validation | Assumed correct | Validates format |
| Consistency checks | None | Assertions for sync |
| Logging | Basic warnings | Structured logging with timestamps |
| Progress tracking | Basic tqdm | Tqdm + detailed summaries |

## Testing Recommendations

To verify the script works correctly:

1. **Check logs for skipped files**:
   ```bash
   python prepare_gigamidi_filtered.py --corpus-dir /data --output-dir /output 2>&1 | grep "Skipped"
   ```

2. **Verify counts match**:
   ```bash
   python -c "
   from lhotse import load_manifest
   recs = load_manifest('/output/gigamidi_recordings_train.jsonl.gz')
   sups = load_manifest('/output/gigamidi_supervisions_train.jsonl.gz')
   print(f'Recordings: {len(recs)}, Supervisions: {len(sups)}')
   assert len(recs) == len(sups), 'Mismatch!'
   print('✓ All counts match!')
   "
   ```

3. **Validate all recordings have supervisions**:
   ```bash
   python -c "
   from lhotse import load_manifest
   recs = load_manifest('/output/gigamidi_recordings_train.jsonl.gz')
   sups = load_manifest('/output/gigamidi_supervisions_train.jsonl.gz')
   rec_ids = set(r.id for r in recs)
   sup_ids = set(s.recording_id for s in sups)
   assert rec_ids == sup_ids, 'ID mismatch!'
   print('✓ All recordings have supervisions!')
   "
   ```
