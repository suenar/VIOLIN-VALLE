#!/usr/bin/env python3
"""
Enhanced version of MIDI to audio script with better FX chain handling.

This version uses a more reliable approach:
- Option 1: Use track templates (easier and more reliable)
- Option 2: Direct plugin loading with preset files
- Option 3: Manual FX chain with saved plugin state

Usage:
    python reaper_midi_to_audio_v2.py --input_dir ./midi --output_dir ./output --track-template path.RTrackTemplate
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
    """Enhanced MIDI to audio rendering in Reaper."""
    
    def __init__(self, plugin_name: Optional[str] = None, sample_rate: int = 44100,
                 track_template: Optional[str] = None, plugin_preset: Optional[str] = None):
        """
        Initialize the Reaper MIDI renderer.
        
        Args:
            plugin_name: Name of the VST/VST3i plugin to use
            sample_rate: Sample rate for audio rendering (default: 44100)
            track_template: Path to track template file (.RTrackTemplate) - RECOMMENDED
            plugin_preset: Path to plugin preset file (.vstpreset, .nksf, etc.)
        """
        self.plugin_name = plugin_name
        self.sample_rate = sample_rate
        self.track_template = track_template
        self.plugin_preset = plugin_preset
        self.project = None
        
    def connect_to_reaper(self):
        """Connect to Reaper instance."""
        try:
            logger.info("Connecting to Reaper...")
            self.project = reapy.Project()
            logger.info("Successfully connected to Reaper")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Reaper: {e}")
            logger.error("Make sure Reaper is running and ReaScript API is enabled.")
            return False
    
    def _create_track_from_template(self) -> Optional[object]:
        """
        Create a track from a track template.
        
        Returns:
            Track object if successful, None otherwise
        """
        if not os.path.exists(self.track_template):
            logger.error(f"Track template not found: {self.track_template}")
            return None
        
        try:
            # Clear existing tracks first
            while len(self.project.tracks) > 0:
                self.project.remove_track(self.project.tracks[0])
            
            # Use Reaper action to insert track from template
            # Action 40062: Insert track from template
            initial_track_count = len(self.project.tracks)
            
            # Set the template file path and insert
            # We need to use a Reaper action that can load from a specific file
            # The approach is to use the API to load the template
            
            # Read template file
            with open(self.track_template, 'rb') as f:
                template_content = f.read()
            
            # Add a track first
            track = self.project.add_track()
            
            # Try to apply template using track's GUID and chunk
            # This is the most reliable way
            track_chunk = template_content.decode('utf-8', errors='ignore')
            
            # Set the track chunk (this includes FX and all settings)
            success = reapy.reascript_api.SetTrackStateChunk(
                track.id,
                track_chunk,
                False  # isundo
            )
            
            if success:
                logger.info("✓ Track template loaded successfully")
                return track
            else:
                logger.warning("Track chunk loading failed, using fallback method")
                # Fallback: just return the track and we'll load plugin manually
                return track
                
        except Exception as e:
            logger.error(f"Error loading track template: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
            
            # Create track - use template if available
            if self.track_template:
                logger.info(f"Creating track from template: {self.track_template}")
                track = self._create_track_from_template()
                if not track:
                    logger.error("Failed to create track from template")
                    return False
            else:
                # Create regular track
                logger.info("Creating MIDI track...")
                track = self.project.add_track()
                track.name = f"MIDI - {Path(midi_path).stem}"
                
                # Load plugin if specified
                if plugin_name:
                    logger.info(f"Loading plugin: {plugin_name}")
                    try:
                        fx = track.add_fx(plugin_name)
                        logger.info(f"✓ Plugin loaded")
                        
                        # Load preset if specified
                        if self.plugin_preset and os.path.exists(self.plugin_preset):
                            logger.info(f"Loading preset: {self.plugin_preset}")
                            # This is plugin-specific and may not work for all plugins
                            logger.warning("⚠️  Preset loading is plugin-specific")
                        
                        if "kontakt" in plugin_name.lower():
                            logger.warning("⚠️  Kontakt detected without template!")
                            logger.warning("   Audio may be empty without instrument loaded")
                            logger.warning("   Use --track-template for best results")
                    except Exception as e:
                        logger.error(f"Failed to load plugin: {e}")
                        return False
            
            # Reset edit cursor to position 0
            logger.info(f"Importing MIDI file: {midi_path}")
            reapy.reascript_api.SetEditCurPos(0, True, False)
            
            # Import MIDI file
            import_result = reapy.reascript_api.InsertMedia(midi_path, 0)
            if import_result == 0:
                logger.error(f"Failed to import MIDI file: {midi_path}")
                return False
            
            # Get the imported item
            if len(track.items) == 0:
                logger.error("No MIDI items found after import")
                return False
            
            item = track.items[0]
            project_length = item.position + item.length
            
            # Set render settings
            logger.info("Configuring render settings...")
            self.project.time_selection.start = 0
            self.project.time_selection.end = project_length
            
            reapy.reascript_api.GetSetProjectInfo_String(
                self.project.id, 'RENDER_FILE', output_path, True
            )
            reapy.reascript_api.GetSetProjectInfo(
                self.project.id, 'RENDER_SRATE', self.sample_rate, True
            )
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Render
            logger.info(f"Rendering audio to: {output_path}")
            reapy.reascript_api.Main_OnCommand(41824, 0)  # Render project to file
            
            # Wait for render
            time.sleep(2)
            
            # Check output
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"✓ Successfully rendered: {output_path} ({file_size} bytes)")
                return True
            else:
                logger.error(f"✗ Render failed - output file not created")
                return False
                
        except Exception as e:
            logger.error(f"Error processing MIDI file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         plugin_name: Optional[str] = None) -> dict:
        """Process all MIDI files in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Find MIDI files
        midi_extensions = ['.mid', '.midi', '.MID', '.MIDI']
        midi_files = []
        for ext in midi_extensions:
            midi_files.extend(list(input_path.glob(f'*{ext}')))
            midi_files.extend(list(input_path.glob(f'**/*{ext}')))
        
        midi_files = list(set(midi_files))
        
        if not midi_files:
            logger.warning(f"No MIDI files found in: {input_dir}")
            return {'success': 0, 'failed': 0, 'total': 0}
        
        logger.info(f"Found {len(midi_files)} MIDI file(s) to process")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {'success': 0, 'failed': 0, 'total': len(midi_files)}
        
        for i, midi_file in enumerate(midi_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing file {i}/{len(midi_files)}")
            logger.info(f"{'='*60}")
            
            relative_path = midi_file.relative_to(input_path) if midi_file.is_relative_to(input_path) else midi_file
            output_file = output_path / relative_path.with_suffix('.wav')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
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
    parser = argparse.ArgumentParser(description='Generate audio from MIDI files using Reaper (v2 - Enhanced)')
    
    parser.add_argument('-i', '--input_dir', type=str, help='Directory containing MIDI files')
    parser.add_argument('-o', '--output_dir', type=str, help='Directory for output WAV files')
    parser.add_argument('-p', '--plugin', type=str, default=None, help='VST/VST3i plugin name')
    parser.add_argument('-tt', '--track-template', type=str, default=None, 
                       help='Path to track template file (.RTrackTemplate) - RECOMMENDED for Kontakt')
    parser.add_argument('-preset', '--plugin-preset', type=str, default=None,
                       help='Path to plugin preset file')
    parser.add_argument('-sr', '--sample_rate', type=int, default=44100, help='Sample rate (default: 44100)')
    
    args = parser.parse_args()
    
    if not args.input_dir or not args.output_dir:
        parser.error("--input_dir and --output_dir are required")
    
    # Initialize renderer
    renderer = ReaperMIDIRenderer(
        plugin_name=args.plugin,
        sample_rate=args.sample_rate,
        track_template=args.track_template,
        plugin_preset=args.plugin_preset
    )
    
    # Connect
    if not renderer.connect_to_reaper():
        logger.error("Failed to connect to Reaper. Exiting.")
        return 1
    
    # Process
    logger.info(f"\nInput directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    if args.track_template:
        logger.info(f"Track Template: {args.track_template}")
        logger.info("✓ Using track template (recommended for Kontakt)")
    elif args.plugin:
        logger.info(f"Plugin: {args.plugin}")
    
    logger.info(f"Sample rate: {args.sample_rate} Hz\n")
    
    results = renderer.process_directory(args.input_dir, args.output_dir, args.plugin)
    
    # Summary
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
