#!/usr/bin/env python3
"""
Script to adjust MIDI file tempo to match corresponding audio file duration.

Usage:
    python adjust_midi_tempo.py --midi_dir <midi_directory> --audio_dir <audio_directory> --output_dir <output_directory>
    
    Or for a single pair:
    python adjust_midi_tempo.py --midi <midi_file> --audio <audio_file> --output <output_file>
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import pretty_midi
import librosa
import soundfile as sf


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    # Use soundfile first as it's faster for getting duration
    try:
        info = sf.info(audio_path)
        return info.duration
    except:
        # Fallback to librosa if soundfile fails
        y, sr = librosa.load(audio_path, sr=None)
        return librosa.get_duration(y=y, sr=sr)


def get_midi_duration(midi_path: str) -> float:
    """
    Get duration of MIDI file in seconds.
    
    Args:
        midi_path: Path to MIDI file
        
    Returns:
        Duration in seconds
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    return midi_data.get_end_time()


def adjust_midi_tempo(midi_path: str, target_duration: float, output_path: str) -> Tuple[float, float]:
    """
    Adjust MIDI tempo to match target duration.
    
    Args:
        midi_path: Path to input MIDI file
        target_duration: Target duration in seconds
        output_path: Path to save adjusted MIDI file
        
    Returns:
        Tuple of (original_duration, new_duration)
    """
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    original_duration = midi_data.get_end_time()
    
    if original_duration == 0:
        print(f"Warning: MIDI file {midi_path} has zero duration!")
        return original_duration, original_duration
    
    # Calculate tempo adjustment ratio
    # If we need to make the MIDI longer, we slow it down (decrease tempo)
    # If we need to make it shorter, we speed it up (increase tempo)
    time_stretch_ratio = target_duration / original_duration
    
    # Adjust all note timings
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note.start *= time_stretch_ratio
            note.end *= time_stretch_ratio
        
        # Adjust control changes
        for control_change in instrument.control_changes:
            control_change.time *= time_stretch_ratio
        
        # Adjust pitch bends
        for pitch_bend in instrument.pitch_bends:
            pitch_bend.time *= time_stretch_ratio
    
    # Adjust tempo changes
    # The tempo change times need to be stretched
    if hasattr(midi_data, '_tick_scales'):
        # Adjust tick scales (tempo changes)
        new_tick_scales = []
        for tick, scale in midi_data._tick_scales:
            new_tick = tick * time_stretch_ratio
            new_tick_scales.append((new_tick, scale * time_stretch_ratio))
        midi_data._tick_scales = new_tick_scales
    
    # Alternative approach: adjust tempo directly
    # Get current tempo and scale it
    tempo_times, tempos = midi_data.get_tempo_changes()
    if len(tempos) > 0:
        # Scale all tempos by inverse of time stretch ratio
        # (slower music = lower BPM, faster music = higher BPM)
        for i in range(len(tempos)):
            tempos[i] = tempos[i] / time_stretch_ratio
        
        # Update tempo changes with new timings and tempos
        # This is a workaround since pretty_midi doesn't have a direct tempo setter
        # We'll recreate the MIDI with adjusted timings instead
    
    # Save adjusted MIDI
    midi_data.write(output_path)
    
    # Verify new duration
    new_midi = pretty_midi.PrettyMIDI(output_path)
    new_duration = new_midi.get_end_time()
    
    return original_duration, new_duration


def process_file_pair(midi_path: str, audio_path: str, output_path: str, verbose: bool = True) -> bool:
    """
    Process a single MIDI-audio file pair.
    
    Args:
        midi_path: Path to MIDI file
        audio_path: Path to audio file
        output_path: Path to save adjusted MIDI
        verbose: Whether to print progress
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get durations
        audio_duration = get_audio_duration(audio_path)
        midi_duration = get_midi_duration(midi_path)
        
        if verbose:
            print(f"\nProcessing: {os.path.basename(midi_path)}")
            print(f"  Audio duration: {audio_duration:.2f}s")
            print(f"  MIDI duration:  {midi_duration:.2f}s")
        
        # Adjust MIDI tempo
        original_duration, new_duration = adjust_midi_tempo(midi_path, audio_duration, output_path)
        
        if verbose:
            print(f"  New MIDI duration: {new_duration:.2f}s")
            print(f"  Adjustment ratio: {audio_duration/original_duration:.3f}x")
            print(f"  ✓ Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing {midi_path}: {str(e)}")
        return False


def process_directory(midi_dir: str, audio_dir: str, output_dir: str, verbose: bool = True):
    """
    Process all MIDI files in a directory with corresponding audio files.
    
    Args:
        midi_dir: Directory containing MIDI files
        audio_dir: Directory containing audio files
        output_dir: Directory to save adjusted MIDI files
        verbose: Whether to print progress
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all MIDI files
    midi_extensions = ['.mid', '.midi', '.MID', '.MIDI']
    midi_files = []
    for ext in midi_extensions:
        midi_files.extend(Path(midi_dir).glob(f'*{ext}'))
    
    if not midi_files:
        print(f"No MIDI files found in {midi_dir}")
        return
    
    # Audio extensions to look for
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus']
    
    successful = 0
    failed = 0
    
    for midi_path in sorted(midi_files):
        # Find corresponding audio file
        base_name = midi_path.stem
        audio_path = None
        
        for ext in audio_extensions:
            potential_audio = Path(audio_dir) / f"{base_name}{ext}"
            if potential_audio.exists():
                audio_path = potential_audio
                break
        
        if audio_path is None:
            print(f"✗ No corresponding audio file found for {midi_path.name}")
            failed += 1
            continue
        
        # Output path
        output_path = Path(output_dir) / midi_path.name
        
        # Process the pair
        if process_file_pair(str(midi_path), str(audio_path), str(output_path), verbose):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Adjust MIDI file tempo to match corresponding audio file duration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file pair
  python adjust_midi_tempo.py --midi song.mid --audio song.wav --output song_adjusted.mid
  
  # Process entire directories
  python adjust_midi_tempo.py --midi_dir ./midi_files --audio_dir ./audio_files --output_dir ./adjusted_midi
        """
    )
    
    # Single file mode
    parser.add_argument('--midi', type=str, help='Path to single MIDI file')
    parser.add_argument('--audio', type=str, help='Path to single audio file')
    parser.add_argument('--output', type=str, help='Path to output MIDI file')
    
    # Directory mode
    parser.add_argument('--midi_dir', type=str, help='Directory containing MIDI files')
    parser.add_argument('--audio_dir', type=str, help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, help='Directory to save adjusted MIDI files')
    
    # Options
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Check if single file mode or directory mode
    single_mode = args.midi and args.audio and args.output
    dir_mode = args.midi_dir and args.audio_dir and args.output_dir
    
    if not single_mode and not dir_mode:
        parser.error("Either provide (--midi, --audio, --output) for single file mode, "
                    "or (--midi_dir, --audio_dir, --output_dir) for directory mode")
    
    if single_mode and dir_mode:
        parser.error("Cannot use both single file mode and directory mode simultaneously")
    
    verbose = not args.quiet
    
    if single_mode:
        # Process single file pair
        success = process_file_pair(args.midi, args.audio, args.output, verbose)
        if not success:
            exit(1)
    else:
        # Process directory
        process_directory(args.midi_dir, args.audio_dir, args.output_dir, verbose)


if __name__ == "__main__":
    main()
