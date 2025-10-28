#!/usr/bin/env python3
"""
Script to generate violin audio from MIDI files using Reaper's Python API.

This script:
1. Loads MIDI files from a specified directory
2. Applies a VST/VST3i plugin (violin instrument)
3. Renders the audio to WAV files
4. Saves the output files

Requirements:
    - Reaper DAW installed and running
    - reapy library: pip install python-reapy
    - VST/VST3i plugins installed

Usage:
    python reaper_midi_to_audio.py --input_dir /path/to/midi/files \\
                                   --output_dir /path/to/output \\
                                   --plugin "VST3: Violin Plugin Name"
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional, List
import logging

try:
    import reapy
except ImportError:
    print("ERROR: reapy library not found. Please install it with: pip install python-reapy")
    exit(1)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReaperMIDIRenderer:
    """Handles MIDI to audio rendering in Reaper."""
    
    def __init__(self, plugin_name: Optional[str] = None, sample_rate: int = 44100):
        """
        Initialize the Reaper MIDI renderer.
        
        Args:
            plugin_name: Name of the VST/VST3i plugin to use (e.g., "VST3: Kontakt 7")
            sample_rate: Sample rate for audio rendering (default: 44100)
        """
        self.plugin_name = plugin_name
        self.sample_rate = sample_rate
        self.project = None
        
    def connect_to_reaper(self):
        """Connect to Reaper instance."""
        try:
            # Check if Reaper is running and enable ReaScript API
            logger.info("Connecting to Reaper...")
            self.project = reapy.Project()
            logger.info("Successfully connected to Reaper")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Reaper: {e}")
            logger.error("Make sure Reaper is running and ReaScript API is enabled.")
            logger.error("In Reaper: Options -> Preferences -> Plug-ins -> ReaScript -> Enable Python for use with ReaScript")
            return False
    
    def list_available_plugins(self) -> List[str]:
        """List all available VST/VST3i plugins in Reaper."""
        logger.info("Scanning available plugins...")
        plugins = []
        
        try:
            # Create a temporary track to scan plugins
            track = self.project.add_track()
            
            # Get FX list - this varies by system
            # You can also use RPR_EnumInstalledFX to list installed plugins
            logger.info("Plugins can be listed using Reaper's FX Browser")
            logger.info("Common plugin name formats:")
            logger.info("  - VST3: Plugin Name (Manufacturer)")
            logger.info("  - VST: Plugin Name (Manufacturer)")
            logger.info("  - VSTi: Plugin Name")
            
            # Clean up
            self.project.remove_track(track)
            
        except Exception as e:
            logger.warning(f"Could not enumerate plugins: {e}")
        
        return plugins
    
    def process_midi_file(self, midi_path: str, output_path: str, 
                         plugin_name: Optional[str] = None) -> bool:
        """
        Process a single MIDI file and render to audio.
        
        Args:
            midi_path: Path to input MIDI file
            output_path: Path for output WAV file
            plugin_name: Override plugin name for this file
            
        Returns:
            True if successful, False otherwise
        """
        if plugin_name is None:
            plugin_name = self.plugin_name
            
        logger.info(f"Processing: {midi_path}")
        
        try:
            # Clear existing tracks
            logger.info("Clearing project...")
            while len(self.project.tracks) > 0:
                self.project.remove_track(self.project.tracks[0])
            
            # Add a new track
            logger.info("Creating MIDI track...")
            track = self.project.add_track()
            track.name = f"MIDI - {Path(midi_path).stem}"
            
            # Reset edit cursor to position 0 to ensure MIDI imports at start
            logger.info(f"Importing MIDI file: {midi_path}")
            reapy.reascript_api.SetEditCurPos(0, True, False)  # Set cursor to 0
            
            # Import MIDI file at cursor position (now at 0)
            import_result = reapy.reascript_api.InsertMedia(midi_path, 0)
            if import_result == 0:
                logger.error(f"Failed to import MIDI file: {midi_path}")
                return False
            
            # Get the imported item and adjust track
            if len(track.items) == 0:
                logger.error("No MIDI items found after import")
                return False
            
            item = track.items[0]
            
            # Load VST/VST3i plugin on the track
            if plugin_name:
                logger.info(f"Loading plugin: {plugin_name}")
                try:
                    fx = track.add_fx(plugin_name)
                    logger.info(f"Plugin loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load plugin '{plugin_name}': {e}")
                    logger.error("Available plugin name formats:")
                    logger.error("  - 'VST3: Plugin Name (Manufacturer)'")
                    logger.error("  - 'VST: Plugin Name (Manufacturer)'")
                    logger.error("  - 'VSTi: Plugin Name'")
                    logger.error("\nTip: Open Reaper's FX Browser to see exact plugin names")
                    return False
            else:
                logger.warning("No plugin specified - using default MIDI playback")
            
            # Calculate project length based on MIDI content
            project_length = item.position + item.length
            
            # Set project render settings
            logger.info("Configuring render settings...")
            
            # Set render bounds (time selection)
            self.project.time_selection.start = 0
            self.project.time_selection.end = project_length
            
            # Configure render settings using ReaScript API
            reapy.reascript_api.GetSetProjectInfo_String(
                self.project.id, 'RENDER_FILE', output_path, True
            )
            reapy.reascript_api.GetSetProjectInfo_String(
                self.project.id, 'RENDER_PATTERN', '', True
            )
            
            # Set render format to WAV (0x464C4157 = 'WAVE' in hex)
            reapy.reascript_api.GetSetProjectInfo(
                self.project.id, 'RENDER_FORMAT', 0x20000000, True  # WAV format base
            )
            
            # Set sample rate
            reapy.reascript_api.GetSetProjectInfo(
                self.project.id, 'RENDER_SRATE', self.sample_rate, True
            )
            
            # Set to render to file
            reapy.reascript_api.GetSetProjectInfo(
                self.project.id, 'RENDER_TAILFLAG', 1, True  # Include tail
            )
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Render the project
            logger.info(f"Rendering audio to: {output_path}")
            logger.info("This may take a while depending on the MIDI file length...")
            
            # Render using time selection
            reapy.reascript_api.Main_OnCommand(41824, 0)  # Render project to file
            
            # Wait for render to complete
            # Note: In practice, you may need to poll or use callbacks
            time.sleep(2)  # Basic wait - adjust as needed
            
            # Check if output file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"✓ Successfully rendered: {output_path} ({file_size} bytes)")
                return True
            else:
                logger.error(f"✗ Render failed - output file not created: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing MIDI file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         plugin_name: Optional[str] = None) -> dict:
        """
        Process all MIDI files in a directory.
        
        Args:
            input_dir: Directory containing MIDI files
            output_dir: Directory for output WAV files
            plugin_name: Plugin to use for rendering
            
        Returns:
            Dictionary with success/failure counts
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Find all MIDI files
        midi_extensions = ['.mid', '.midi', '.MID', '.MIDI']
        midi_files = []
        for ext in midi_extensions:
            midi_files.extend(list(input_path.glob(f'*{ext}')))
            midi_files.extend(list(input_path.glob(f'**/*{ext}')))
        
        # Remove duplicates
        midi_files = list(set(midi_files))
        
        if not midi_files:
            logger.warning(f"No MIDI files found in: {input_dir}")
            return {'success': 0, 'failed': 0, 'total': 0}
        
        logger.info(f"Found {len(midi_files)} MIDI file(s) to process")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each MIDI file
        results = {'success': 0, 'failed': 0, 'total': len(midi_files)}
        
        for i, midi_file in enumerate(midi_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing file {i}/{len(midi_files)}")
            logger.info(f"{'='*60}")
            
            # Generate output filename
            relative_path = midi_file.relative_to(input_path) if midi_file.is_relative_to(input_path) else midi_file
            output_file = output_path / relative_path.with_suffix('.wav')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Process the file
            success = self.process_midi_file(
                str(midi_file),
                str(output_file),
                plugin_name
            )
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
        
        return results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate violin audio from MIDI files using Reaper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process MIDI files with a specific violin plugin
  python reaper_midi_to_audio.py -i ./midi_files -o ./audio_output \\
      -p "VST3: Kontakt 7 (Native Instruments)"
  
  # Process with default MIDI synth (no plugin)
  python reaper_midi_to_audio.py -i ./midi_files -o ./audio_output
  
  # List plugin format help
  python reaper_midi_to_audio.py --list-plugins

Note: Reaper must be running before executing this script!
        """
    )
    
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        help='Directory containing MIDI files to process'
    )
    
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        help='Directory for output WAV files'
    )
    
    parser.add_argument(
        '-p', '--plugin',
        type=str,
        default=None,
        help='VST/VST3i plugin name (e.g., "VST3: Kontakt 7", "VSTi: Spitfire Audio")'
    )
    
    parser.add_argument(
        '-sr', '--sample_rate',
        type=int,
        default=44100,
        help='Sample rate for output audio (default: 44100)'
    )
    
    parser.add_argument(
        '--list-plugins',
        action='store_true',
        help='Show information about plugin naming format and exit'
    )
    
    args = parser.parse_args()
    
    # Handle --list-plugins
    if args.list_plugins:
        print("\n" + "="*60)
        print("REAPER PLUGIN NAMING GUIDE")
        print("="*60)
        print("\nTo find the exact plugin name:")
        print("1. Open Reaper")
        print("2. Add a track")
        print("3. Click 'FX' button on the track")
        print("4. Browse for your violin plugin")
        print("5. The exact name shown in the FX browser is what you need")
        print("\nCommon formats:")
        print("  VST3: Plugin Name (Manufacturer)")
        print("  VST: Plugin Name (Manufacturer)")
        print("  VSTi: Plugin Name")
        print("\nExample plugin names:")
        print("  - 'VST3: Kontakt 7 (Native Instruments)'")
        print("  - 'VST: Spitfire Audio Violin (Spitfire Audio)'")
        print("  - 'VSTi: Halion Sonic SE'")
        print("  - 'VST3: BBC Symphony Orchestra (Spitfire Audio)'")
        print("\nTip: Copy the exact name from Reaper's FX browser")
        print("="*60 + "\n")
        return
    
    # Validate required arguments
    if not args.input_dir or not args.output_dir:
        parser.error("--input_dir and --output_dir are required (or use --list-plugins)")
    
    # Initialize renderer
    renderer = ReaperMIDIRenderer(
        plugin_name=args.plugin,
        sample_rate=args.sample_rate
    )
    
    # Connect to Reaper
    if not renderer.connect_to_reaper():
        logger.error("Failed to connect to Reaper. Exiting.")
        return 1
    
    # Process directory
    logger.info(f"\nInput directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    if args.plugin:
        logger.info(f"Plugin: {args.plugin}")
    else:
        logger.warning("No plugin specified - will use default MIDI synth")
    logger.info(f"Sample rate: {args.sample_rate} Hz\n")
    
    # Process all files
    results = renderer.process_directory(
        args.input_dir,
        args.output_dir,
        args.plugin
    )
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total files: {results['total']}")
    logger.info(f"Successfully processed: {results['success']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info("="*60 + "\n")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == '__main__':
    exit(main())
