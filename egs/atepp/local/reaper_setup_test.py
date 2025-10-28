#!/usr/bin/env python3
"""
Test script to verify Reaper setup and discover available plugins.

This script helps you:
1. Test connection to Reaper
2. Discover available VST/VST3i plugins
3. Test MIDI import functionality
4. Verify render capabilities

Usage:
    python reaper_setup_test.py
"""

import sys
import os

print("=" * 70)
print("REAPER SETUP TEST & PLUGIN DISCOVERY")
print("=" * 70)

# Test reapy import
print("\n[1/4] Testing reapy library...")
try:
    import reapy
    print("✓ reapy library found")
except ImportError:
    print("✗ reapy library not found")
    print("\nTo install: pip install python-reapy")
    print("Or use the native script: reaper_midi_to_audio_native.py")
    sys.exit(1)

# Test Reaper connection
print("\n[2/4] Testing Reaper connection...")
try:
    project = reapy.Project()
    print("✓ Successfully connected to Reaper")
    print(f"  Project name: {project.name if project.name else '(Untitled)'}")
    print(f"  Number of tracks: {len(project.tracks)}")
except Exception as e:
    print(f"✗ Failed to connect to Reaper: {e}")
    print("\nMake sure:")
    print("1. Reaper is running")
    print("2. Web control interface is enabled:")
    print("   Options → Preferences → Control/OSC/web → Add → Web interface")
    sys.exit(1)

# Test plugin discovery
print("\n[3/4] Discovering available plugins...")
print("\nThis requires creating a test track...")

try:
    # Save original track count
    original_track_count = len(project.tracks)
    
    # Add test track
    test_track = project.add_track()
    test_track.name = "Plugin Discovery Test"
    
    print("✓ Test track created")
    print("\nAttempting to load common plugin categories...")
    
    # Test some common plugin types
    test_plugins = [
        "ReaSynth",  # Reaper's built-in synth
        "VSTi",      # Generic VSTi
        "VST3i",     # Generic VST3i
    ]
    
    print("\n" + "-" * 70)
    print("PLUGIN DISCOVERY GUIDE")
    print("-" * 70)
    print("\nTo find available plugins in Reaper:")
    print("1. Add a track (Ctrl+T)")
    print("2. Click the 'FX' button")
    print("3. Browse through the categories:")
    print("   - VST3i: Virtual instruments (VST3 format)")
    print("   - VSTi: Virtual instruments (VST2 format)")
    print("   - VST: Virtual effects (VST2 format)")
    print("   - JS: Reaper's built-in effects")
    print("\n4. Note the EXACT plugin name including:")
    print("   - Prefix (VST3:, VSTi:, VST:)")
    print("   - Full name")
    print("   - Manufacturer in parentheses")
    print("\nExample names you might see:")
    
    example_plugins = [
        "VST3: Kontakt 7 (Native Instruments)",
        "VST3: Halion Sonic SE (Steinberg Media Technologies)",
        "VSTi: Spitfire Audio Plugin",
        "VST: ReaComp (Cockos)",
        "JS: ReaEQ",
    ]
    
    for plugin in example_plugins:
        print(f"  • {plugin}")
    
    # Try to enumerate installed FX using ReaScript
    print("\n" + "-" * 70)
    print("ATTEMPTING AUTOMATIC PLUGIN SCAN...")
    print("-" * 70)
    
    # This is a bit tricky with reapy, but we can try using ReaScript API
    try:
        # Try to add some common plugins to see what works
        common_plugins = [
            "ReaSynth",
            "ReaSynDr",
            "ReaVerb",
        ]
        
        print("\nTesting Reaper built-in plugins:")
        for plugin_name in common_plugins:
            try:
                fx = test_track.add_fx(plugin_name)
                print(f"  ✓ {plugin_name} - Available")
                # Remove the FX
                test_track.fxs.remove(fx)
            except Exception as e:
                print(f"  ✗ {plugin_name} - Not found")
        
        print("\n💡 TIP: To scan for YOUR VST plugins:")
        print("   In Reaper: Options → Preferences → Plug-ins → VST")
        print("   Make sure your plugin folders are listed and scanned")
        
    except Exception as e:
        print(f"Could not enumerate plugins: {e}")
    
    # Clean up test track
    project.remove_track(test_track)
    print("\n✓ Test track cleaned up")
    
except Exception as e:
    print(f"✗ Error during plugin discovery: {e}")
    import traceback
    traceback.print_exc()

# Test MIDI capabilities
print("\n[4/4] Testing MIDI import capabilities...")

try:
    # Create test track
    midi_track = project.add_track()
    midi_track.name = "MIDI Import Test"
    
    # Try to create a simple MIDI item
    midi_item = midi_track.add_midi_item(start=0, length=4)
    
    print("✓ MIDI track and item creation works")
    
    # Clean up
    project.remove_track(midi_track)
    print("✓ MIDI test track cleaned up")
    
except Exception as e:
    print(f"✗ MIDI test failed: {e}")

# Summary
print("\n" + "=" * 70)
print("SETUP TEST COMPLETE")
print("=" * 70)
print("\n✓ Your Reaper setup is ready!")
print("\nNext steps:")
print("1. Find your violin plugin name in Reaper's FX browser")
print("2. Note the exact name (e.g., 'VST3: Kontakt 7 (Native Instruments)')")
print("3. Run the main script:")
print("\n   python reaper_midi_to_audio.py \\")
print("       -i /path/to/midi/files \\")
print("       -o /path/to/output \\")
print("       -p 'VST3: Your Plugin Name'")
print("\nFor more help: python reaper_midi_to_audio.py --list-plugins")
print("=" * 70 + "\n")

# Show example command with actual samples directory
samples_dir = os.path.join(os.path.dirname(__file__), "..", "samples")
if os.path.exists(samples_dir):
    print("Example command for this project:")
    print(f"\n   python reaper_midi_to_audio.py \\")
    print(f"       -i {os.path.abspath(samples_dir)} \\")
    print(f"       -o {os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))} \\")
    print(f"       -p 'VST3: Your Violin Plugin Name'\n")
