"""
This file was developed as the lhotse receipt for preparing the data.
Modified to filter out recordings without transcriptions.
"""

import logging
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

def prepare_gigamidi(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None, tokenizer: str = "expression"
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param tokenizer: str, tokenizer type ("expression" or other)
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    transcript_dir = corpus_dir / "midi_seg"
    assert transcript_dir.is_dir(), f"No such directory: {transcript_dir}"
    
    # Build transcript dictionary with validation
    if tokenizer == "expression":
        transcript_files = transcript_dir.rglob("**/*.npy")
        transcript_files = [file for file in transcript_files if 'remi' not in str(file)]
    else:
        transcript_files = transcript_dir.rglob("**/*_remi.npy")
    
    transcript_dict = {}
    invalid_transcripts = []
    
    for file in transcript_files:
        # Verify the transcript file actually exists and is readable
        if not file.is_file():
            logging.warning(f"Transcript file does not exist: {file}")
            invalid_transcripts.append(str(file))
            continue
            
        if tokenizer == "expression":
            transcript_dict[file.stem] = str(file)
        else:
            transcript_dict[file.stem.split("_remi")[0]] = str(file)
    
    logging.info(f"Found {len(transcript_dict)} valid transcripts")
    if invalid_transcripts:
        logging.warning(f"Found {len(invalid_transcripts)} invalid transcript files")
    
    manifests = defaultdict(dict)
    dataset_parts = ["test", "validation", "train"]
    
    for part in tqdm(
        dataset_parts,
        desc="Process gigamidi audio and text, it takes about 160 seconds.",
    ):
        logging.info(f"Processing gigamidi subset: {part}")
        
        recordings = []
        supervisions = []
        skipped_no_transcript = []
        skipped_invalid_audio = []
        skipped_zero_duration = []
        
        wav_path = corpus_dir / "wav_seg" / f"{part}"
        
        if not wav_path.exists():
            logging.warning(f"Audio directory does not exist: {wav_path}")
            continue
        
        audio_files = list(wav_path.rglob("**/*.wav"))
        logging.info(f"Found {len(audio_files)} audio files in {part}")
        
        for audio_path in tqdm(audio_files, desc=f"Processing {part}"):
            idx = audio_path.stem
            
            # Extract speaker and language from filename
            parts = audio_path.stem.split("_")
            if len(parts) < 2:
                logging.warning(f"Invalid filename format (expected speaker_language_...): {audio_path}")
                skipped_invalid_audio.append(str(audio_path))
                continue
                
            speaker = parts[0]
            language = parts[1]
            
            # Check 1: Does transcript exist in our dictionary?
            if idx not in transcript_dict:
                logging.warning(f"No transcript found for: {idx}")
                skipped_no_transcript.append(idx)
                continue
            
            # Check 2: Verify transcript file path exists
            transcript_path = Path(transcript_dict[idx])
            if not transcript_path.is_file():
                logging.warning(f"Transcript file missing: {transcript_path}")
                skipped_no_transcript.append(idx)
                # Remove from dict to prevent future issues
                del transcript_dict[idx]
                continue
            
            # Check 3: Does audio file exist?
            if not audio_path.is_file():
                logging.warning(f"Audio file does not exist: {audio_path}")
                skipped_invalid_audio.append(str(audio_path))
                continue
            
            # Check 4: Create recording and validate duration
            try:
                recording = Recording.from_file(audio_path)
            except Exception as e:
                logging.error(f"Failed to load recording {audio_path}: {e}")
                skipped_invalid_audio.append(str(audio_path))
                continue
            
            if recording.duration <= 0:
                logging.warning(f"Zero duration audio: {audio_path}")
                skipped_zero_duration.append(str(audio_path))
                continue
            
            # All checks passed - add recording and supervision
            recordings.append(recording)
            
            segment = SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=recording.duration,
                channel=0,
                speaker=speaker,
                language=language,
                text=str(transcript_path),  # Store path to transcript
            )
            supervisions.append(segment)
        
        # Log statistics
        logging.info(f"=== {part.upper()} Statistics ===")
        logging.info(f"Valid recordings: {len(recordings)}")
        logging.info(f"Valid supervisions: {len(supervisions)}")
        logging.info(f"Skipped (no transcript): {len(skipped_no_transcript)}")
        logging.info(f"Skipped (invalid audio): {len(skipped_invalid_audio)}")
        logging.info(f"Skipped (zero duration): {len(skipped_zero_duration)}")
        
        # Create sets only if we have valid data
        if len(recordings) == 0:
            logging.warning(f"No valid recordings found for {part}, skipping...")
            continue
        
        # Ensure recordings and supervisions match
        assert len(recordings) == len(supervisions), \
            f"Mismatch: {len(recordings)} recordings vs {len(supervisions)} supervisions"
        
        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        
        # Fix and validate manifests
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)
        
        logging.info(f"Final {part}: {len(recording_set)} recordings, {len(supervision_set)} supervisions")

        # Save manifests
        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"gigamidi_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"gigamidi_recordings_{part}.jsonl.gz")
            logging.info(f"Saved manifests to {output_dir}")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Prepare the gigamidi dataset manifests with filtering."
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        required=True,
        help="Path to the gigamidi corpus directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to the output directory for manifests.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="expression",
        choices=["expression", "remi"],
        help="Tokenizer type to use.",
    )
    
    args = parser.parse_args()
    
    logging.info(f"Starting gigamidi preparation with tokenizer: {args.tokenizer}")
    
    manifests = prepare_gigamidi(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        tokenizer=args.tokenizer,
    )
    
    # Print final summary
    logging.info("=== FINAL SUMMARY ===")
    for part, manifest in manifests.items():
        logging.info(f"{part}: {len(manifest['recordings'])} recordings, "
                    f"{len(manifest['supervisions'])} supervisions")
