#!/usr/bin/env python3
"""
Script to generate violin audio from MIDI files using Reaper's native ReaScript API.

This version uses Reaper's built-in Python ReaScript API directly without external dependencies.
To use this script, you must run it FROM WITHIN Reaper's ReaScript environment.

Usage:
    1. Open Reaper
    2. Go to: Actions -> Show action list
    3. Click "New action" -> "Load ReaScript"
    4. Select this script
    5. Edit the configuration section below and run the script

Or run from command line (if Reaper Python is in PATH):
    python reaper_midi_to_audio_native.py
"""

import os
import sys
from pathlib import Path
import time

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================
INPUT_DIR = "/path/to/midi/files"  # Directory containing MIDI files
OUTPUT_DIR = "/path/to/output"     # Directory for output WAV files
PLUGIN_NAME = "VST3: Violin Plugin"  # Plugin to use (or None for default MIDI)
SAMPLE_RATE = 44100                # Sample rate for output
# ============================================================================

try:
    from reaper_python import *
except ImportError:
    print("ERROR: This script must be run from within Reaper's ReaScript environment")
    print("\nTo run this script:")
    print("1. Open Reaper")
    print("2. Actions -> Show action list -> New action -> Load ReaScript")
    print("3. Select this script file")
    print("\nAlternatively, use reaper_midi_to_audio.py with reapy library")
    sys.exit(1)


def get_midi_files(directory):
    """Get all MIDI files from directory."""
    midi_files = []
    extensions = ['.mid', '.midi', '.MID', '.MIDI']
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                midi_files.append(os.path.join(root, file))
    
    return sorted(midi_files)


def clear_project():
    """Remove all tracks from current project."""
    while RPR_CountTracks(0) > 0:
        track = RPR_GetTrack(0, 0)
        RPR_DeleteTrack(track)
    RPR_UpdateArrange()


def import_midi_to_track(midi_path, track):
    """
    Import MIDI file to a track.
    
    Args:
        midi_path: Path to MIDI file
        track: Reaper track object
    """
    # Insert MIDI file at position 0 on the track
    RPR_InsertMedia(midi_path, 0)  # Mode 0 = insert as new items on existing tracks
    RPR_UpdateArrange()
    
    # Get the newly created item
    num_items = RPR_CountTrackMediaItems(track)
    if num_items > 0:
        return RPR_GetTrackMediaItem(track, num_items - 1)
    return None


def add_plugin_to_track(track, plugin_name):
    """
    Add VST/VST3i plugin to track.
    
    Args:
        track: Reaper track object
        plugin_name: Name of plugin to add
        
    Returns:
        True if successful, False otherwise
    """
    if not plugin_name:
        return True
    
    # Add FX to track
    fx_index = RPR_TrackFX_AddByName(track, plugin_name, False, -1)
    
    if fx_index < 0:
        print(f"ERROR: Failed to add plugin '{plugin_name}'")
        print("Available formats:")
        print("  - VST3: Plugin Name (Manufacturer)")
        print("  - VST: Plugin Name (Manufacturer)")
        print("  - VSTi: Plugin Name")
        return False
    
    print(f"✓ Plugin added successfully at FX slot {fx_index}")
    return True


def render_project(output_path, sample_rate=44100):
    """
    Render current project to WAV file.
    
    Args:
        output_path: Path for output WAV file
        sample_rate: Sample rate for rendering
    """
    project = RPR_EnumProjects(-1, "", 0)[0]
    
    # Set render file path
    RPR_GetSetProjectInfo_String(project, "RENDER_FILE", output_path, True)
    
    # Set render format to WAV
    # Format string: "evaw" = wave format
    RPR_GetSetProjectInfo_String(project, "RENDER_FORMAT", "evaw", True)
    
    # Set sample rate
    RPR_GetSetProjectInfo(project, "RENDER_SRATE", sample_rate, True)
    
    # Set render bounds to entire project
    RPR_GetSetProjectInfo(project, "RENDER_STARTPOS", 0.0, True)
    
    # Get project length
    project_length = RPR_GetProjectLength(project)
    RPR_GetSetProjectInfo(project, "RENDER_ENDPOS", project_length, True)
    
    # Set to render selected items or entire project
    RPR_GetSetProjectInfo(project, "RENDER_SETTINGS", 0, True)  # Render entire project
    
    # Execute render command
    RPR_Main_OnCommand(41824, 0)  # File: Render project to file
    
    # Wait for render to complete
    # Note: This is a simple wait. For production, you'd want to poll render status
    time.sleep(1)
    
    return os.path.exists(output_path)


def process_midi_file(midi_path, output_path, plugin_name=None, sample_rate=44100):
    """
    Process a single MIDI file.
    
    Args:
        midi_path: Path to input MIDI file
        output_path: Path for output WAV file
        plugin_name: VST/VST3i plugin name
        sample_rate: Sample rate for rendering
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\nProcessing: {midi_path}")
    
    try:
        # Clear project
        clear_project()
        
        # Add new track
        RPR_InsertTrackAtIndex(0, True)
        track = RPR_GetTrack(0, 0)
        
        # Set track name
        track_name = f"MIDI - {Path(midi_path).stem}"
        RPR_GetSetMediaTrackInfo_String(track, "P_NAME", track_name, True)
        
        # Import MIDI file
        print(f"  Importing MIDI...")
        item = import_midi_to_track(midi_path, track)
        
        if not item:
            print(f"  ERROR: Failed to import MIDI file")
            return False
        
        # Add plugin
        if plugin_name:
            print(f"  Loading plugin: {plugin_name}")
            if not add_plugin_to_track(track, plugin_name):
                return False
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Render
        print(f"  Rendering to: {output_path}")
        success = render_project(output_path, sample_rate)
        
        if success:
            file_size = os.path.getsize(output_path)
            print(f"  ✓ Success! ({file_size:,} bytes)")
            return True
        else:
            print(f"  ✗ Render failed")
            return False
            
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_directory(input_dir, output_dir, plugin_name=None, sample_rate=44100):
    """
    Process all MIDI files in directory.
    
    Args:
        input_dir: Input directory with MIDI files
        output_dir: Output directory for WAV files
        plugin_name: Plugin to use
        sample_rate: Sample rate for rendering
    """
    # Find MIDI files
    midi_files = get_midi_files(input_dir)
    
    if not midi_files:
        print(f"No MIDI files found in: {input_dir}")
        return
    
    print(f"\nFound {len(midi_files)} MIDI file(s)")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if plugin_name:
        print(f"Plugin: {plugin_name}")
    else:
        print("Plugin: (default MIDI synth)")
    print(f"Sample rate: {sample_rate} Hz")
    print("=" * 60)
    
    # Process each file
    success_count = 0
    failed_count = 0
    
    for i, midi_file in enumerate(midi_files, 1):
        print(f"\n[{i}/{len(midi_files)}]", "=" * 50)
        
        # Generate output path
        rel_path = os.path.relpath(midi_file, input_dir)
        output_file = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.wav')
        
        # Process
        if process_midi_file(midi_file, output_file, plugin_name, sample_rate):
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total files: {len(midi_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print("=" * 60)


def main():
    """Main entry point."""
    print("=" * 60)
    print("REAPER MIDI TO AUDIO RENDERER (Native ReaScript)")
    print("=" * 60)
    
    # Validate configuration
    if INPUT_DIR == "/path/to/midi/files":
        print("\nERROR: Please edit the configuration section at the top of this script")
        print("Set INPUT_DIR, OUTPUT_DIR, and PLUGIN_NAME")
        return
    
    if not os.path.exists(INPUT_DIR):
        print(f"\nERROR: Input directory not found: {INPUT_DIR}")
        return
    
    # Process directory
    process_directory(INPUT_DIR, OUTPUT_DIR, PLUGIN_NAME, SAMPLE_RATE)


# Run main function
if __name__ == "__main__":
    main()
elif __name__ == "__reaper_python__":
    # Running from Reaper's ReaScript environment
    main()
