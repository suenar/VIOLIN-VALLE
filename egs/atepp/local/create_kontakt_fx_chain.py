#!/usr/bin/env python3
"""
Helper script to create and manage FX chains for Kontakt 8 (or any VST with instruments).

This script provides an interactive workflow to:
1. Load Kontakt 8
2. Guide you through selecting an instrument
3. Save the configuration as an FX chain
4. Test the FX chain

Usage:
    python create_kontakt_fx_chain.py
"""

import os
import sys
from pathlib import Path

try:
    import reapy
except ImportError:
    print("ERROR: reapy library not found. Please install it with: pip install python-reapy")
    sys.exit(1)


def get_reaper_fx_chains_directory():
    """Get the default FX chains directory for the current OS."""
    home = Path.home()
    
    if sys.platform == "win32":
        fx_dir = home / "AppData" / "Roaming" / "REAPER" / "FXChains"
    elif sys.platform == "darwin":  # macOS
        fx_dir = home / "Library" / "Application Support" / "REAPER" / "FXChains"
    else:  # Linux
        fx_dir = home / ".config" / "REAPER" / "FXChains"
    
    return fx_dir


def main():
    print("=" * 70)
    print("KONTAKT 8 FX CHAIN CREATOR")
    print("=" * 70)
    print("\nThis helper will guide you through creating an FX chain with")
    print("Kontakt 8 and your chosen violin instrument.\n")
    
    # Connect to Reaper
    print("Step 1: Connecting to Reaper...")
    try:
        project = reapy.Project()
        print("✓ Connected successfully\n")
    except Exception as e:
        print(f"✗ Failed to connect to Reaper: {e}")
        print("\nMake sure:")
        print("1. Reaper is running")
        print("2. Web control is enabled (Options → Preferences → Control/OSC/web)")
        return 1
    
    # Get plugin name
    print("-" * 70)
    print("Step 2: Plugin Name")
    print("-" * 70)
    plugin_name = input("Enter plugin name [VST3: Kontakt 8 (Native Instruments)]: ").strip()
    if not plugin_name:
        plugin_name = "VST3: Kontakt 8 (Native Instruments)"
    print(f"Using: {plugin_name}\n")
    
    # Create a test track
    print("-" * 70)
    print("Step 3: Creating test track and loading plugin...")
    print("-" * 70)
    
    try:
        track = project.add_track()
        track.name = "Kontakt Setup"
        print("✓ Track created")
        
        # Load plugin
        print(f"Loading {plugin_name}...")
        fx = track.add_fx(plugin_name)
        print("✓ Plugin loaded")
        
        # Open FX window
        print("✓ Opening plugin window...")
        fx.open_ui()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    # Wait for user to load instrument
    print("\n" + "=" * 70)
    print("🎻 ACTION REQUIRED - LOAD YOUR INSTRUMENT IN KONTAKT")
    print("=" * 70)
    print("\nThe Kontakt window should now be open.")
    print("\nPlease:")
    print("1. In Kontakt, click 'Browse' or 'Files'")
    print("2. Navigate to your violin library")
    print("3. Load your desired violin instrument")
    print("4. Configure any settings (velocity layers, expression, etc.)")
    print("5. Test the sound if you like")
    print("\n" + "=" * 70)
    
    input("\nPress Enter when you're done configuring Kontakt...")
    
    # Get FX chain name
    print("\n" + "-" * 70)
    print("Step 4: Save FX Chain")
    print("-" * 70)
    
    chain_name = input("Enter a name for this FX chain [Kontakt_Violin]: ").strip()
    if not chain_name:
        chain_name = "Kontakt_Violin"
    
    # Make sure it ends with .RfxChain
    if not chain_name.endswith(".RfxChain"):
        chain_name += ".RfxChain"
    
    # Get save location
    default_dir = get_reaper_fx_chains_directory()
    print(f"\nDefault FX chains directory: {default_dir}")
    
    save_path = input(f"Save path [{default_dir / chain_name}]: ").strip()
    if not save_path:
        save_path = default_dir / chain_name
    else:
        save_path = Path(save_path)
    
    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save using ReaScript API
    print(f"\nSaving FX chain to: {save_path}")
    
    try:
        # The recommended way is to manually save, but we can guide through it
        print("\n⚠️  MANUAL SAVE REQUIRED:")
        print("-" * 70)
        print("1. In the Kontakt FX window, look for the menu (usually '+' icon)")
        print("2. Click it and select 'Save FX chain...' or 'Save track FX chain...'")
        print(f"3. Save as: {chain_name}")
        print(f"4. To directory: {save_path.parent}")
        print("\nOR simply:")
        print(f"Right-click the track → Track FX → Save chain → {chain_name}")
        print("-" * 70)
        
        input("\nPress Enter after you've saved the FX chain...")
        
        # Verify the file was created
        if save_path.exists():
            print(f"✓ FX chain file found: {save_path}")
        else:
            print(f"⚠️  File not found at expected location: {save_path}")
            print("You may have saved it to a different location.")
            
            # Try to find it
            print("\nSearching for .RfxChain files...")
            for fx_file in default_dir.glob("*.RfxChain"):
                print(f"  Found: {fx_file}")
            
            actual_path = input("\nEnter the actual path to your FX chain file: ").strip()
            if actual_path and os.path.exists(actual_path):
                save_path = Path(actual_path)
                print(f"✓ Using: {save_path}")
            else:
                print("⚠️  Could not locate FX chain file")
                save_path = None
        
    except Exception as e:
        print(f"Error during save: {e}")
        save_path = None
    
    # Clean up test track
    print("\n" + "-" * 70)
    print("Step 5: Cleanup")
    print("-" * 70)
    
    try:
        project.remove_track(track)
        print("✓ Test track removed")
    except Exception as e:
        print(f"⚠️  Could not remove test track: {e}")
    
    # Show usage instructions
    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETE!")
    print("=" * 70)
    
    if save_path and save_path.exists():
        print("\nYour FX chain is ready to use!")
        print("\n📋 USAGE:")
        print("-" * 70)
        print("\npython reaper_midi_to_audio.py \\")
        print(f"    -i /path/to/midi/files \\")
        print(f"    -o /path/to/output \\")
        print(f"    -fx '{save_path}'")
        print("\n" + "-" * 70)
        print("\n💡 TIP: You can create multiple FX chains for different instruments:")
        print("   - Kontakt_Violin_Solo.RfxChain")
        print("   - Kontakt_Violin_Ensemble.RfxChain")
        print("   - Kontakt_Cello.RfxChain")
        print("   etc.")
    else:
        print("\n⚠️  FX chain file location unclear.")
        print("\nManually locate your .RfxChain file and use it with:")
        print("\npython reaper_midi_to_audio.py \\")
        print("    -i /path/to/midi/files \\")
        print("    -o /path/to/output \\")
        print("    -fx '/path/to/your/chain.RfxChain'")
    
    print("\n" + "=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
