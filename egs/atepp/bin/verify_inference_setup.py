#!/usr/bin/env python3
"""
Verification script to check if audio-to-audio inference is properly set up.

Usage:
    python3 bin/verify_inference_setup.py
    python3 bin/verify_inference_setup.py --checkpoint exp/VALLE-audio/checkpoint-20000.pt
"""

import argparse
import os
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_success(msg):
    print(f"{GREEN}✓{RESET} {msg}")


def print_error(msg):
    print(f"{RED}✗{RESET} {msg}")


def print_warning(msg):
    print(f"{YELLOW}⚠{RESET} {msg}")


def print_info(msg):
    print(f"{BLUE}ℹ{RESET} {msg}")


def check_file_exists(filepath, name, required=True):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print_success(f"{name}: {filepath}")
        return True
    else:
        if required:
            print_error(f"{name}: {filepath} (NOT FOUND)")
        else:
            print_warning(f"{name}: {filepath} (NOT FOUND - optional)")
        return False


def check_executable(filepath, name):
    """Check if a file is executable."""
    if os.path.exists(filepath):
        if os.access(filepath, os.X_OK):
            print_success(f"{name} is executable")
            return True
        else:
            print_warning(f"{name} exists but is not executable")
            print_info(f"  Run: chmod +x {filepath}")
            return False
    else:
        print_error(f"{name} not found: {filepath}")
        return False


def check_python_imports():
    """Check if required Python packages are available."""
    print("\nChecking Python dependencies...")
    
    required_packages = [
        'torch',
        'torchaudio',
        'numpy',
    ]
    
    optional_packages = [
        'encodec',
    ]
    
    all_ok = True
    
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"Package '{package}' is installed")
        except ImportError:
            print_error(f"Package '{package}' is NOT installed (REQUIRED)")
            all_ok = False
    
    for package in optional_packages:
        try:
            __import__(package)
            print_success(f"Package '{package}' is installed")
        except ImportError:
            print_warning(f"Package '{package}' is NOT installed (optional but recommended)")
    
    return all_ok


def check_model_file(checkpoint_path):
    """Check if model checkpoint is valid."""
    if not checkpoint_path:
        print_warning("No checkpoint path provided (use --checkpoint to specify)")
        return False
    
    if not os.path.exists(checkpoint_path):
        print_error(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    print_info(f"Checking checkpoint: {checkpoint_path}")
    
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model' in checkpoint:
            print_success("Checkpoint has 'model' key")
            state_dict = checkpoint['model']
        else:
            print_warning("Checkpoint doesn't have 'model' key, assuming whole dict is state_dict")
            state_dict = checkpoint
        
        # Check for key components
        ar_keys = [k for k in state_dict.keys() if 'ar_' in k]
        nar_keys = [k for k in state_dict.keys() if 'nar_' in k]
        
        print_info(f"  Found {len(ar_keys)} AR parameters")
        print_info(f"  Found {len(nar_keys)} NAR parameters")
        
        # Check for audio embeddings
        if any('ar_prompt_embeddings' in k for k in state_dict.keys()):
            print_success("Found ar_prompt_embeddings (correct structure)")
        else:
            print_warning("ar_prompt_embeddings not found in checkpoint")
        
        if any('ar_audio_embedding' in k for k in state_dict.keys()):
            # Check if it's singular (correct) or plural (incorrect)
            audio_emb_keys = [k for k in state_dict.keys() if 'ar_audio_embedding' in k]
            if any('ar_audio_embeddings' in k for k in audio_emb_keys):
                print_error("Found ar_audio_embeddings (plural) - should be singular!")
                print_error("This checkpoint may not be compatible")
            else:
                print_success("Found ar_audio_embedding (singular, correct)")
        
        return True
        
    except Exception as e:
        print_error(f"Error loading checkpoint: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify audio-to-audio inference setup")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint to verify (optional)'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("Audio-to-Audio Inference Setup Verification")
    print("="*60)
    
    # Check scripts
    print("\nChecking inference scripts...")
    script_ok = True
    script_ok &= check_file_exists('bin/infer_audiolm.py', 'Main inference script')
    script_ok &= check_file_exists('bin/generate_audio.sh', 'Wrapper script')
    script_ok &= check_executable('bin/infer_audiolm.py', 'Main inference script')
    script_ok &= check_executable('bin/generate_audio.sh', 'Wrapper script')
    
    # Check model file
    print("\nChecking model implementation...")
    model_ok = check_file_exists('../../valle/models/valle_audio.py', 'VALLE-audio model', required=True)
    
    # Check documentation
    print("\nChecking documentation...")
    doc_ok = True
    doc_ok &= check_file_exists('../../AUDIO_TO_AUDIO_INFERENCE.md', 'Main documentation', required=False)
    doc_ok &= check_file_exists('../../QUICK_REFERENCE_INFERENCE.md', 'Quick reference', required=False)
    
    # Check Python dependencies
    deps_ok = check_python_imports()
    
    # Check checkpoint if provided
    checkpoint_ok = True
    if args.checkpoint:
        print("\nChecking model checkpoint...")
        checkpoint_ok = check_model_file(args.checkpoint)
    
    # Check for example prompts
    print("\nChecking for example audio prompts...")
    prompt_dir = Path('prompts')
    if prompt_dir.exists():
        wav_files = list(prompt_dir.glob('*.wav')) + list(prompt_dir.glob('*.mp3'))
        if wav_files:
            print_success(f"Found {len(wav_files)} audio files in prompts/")
            for f in wav_files[:3]:  # Show first 3
                print_info(f"  - {f.name}")
            if len(wav_files) > 3:
                print_info(f"  ... and {len(wav_files) - 3} more")
        else:
            print_warning("No audio files found in prompts/")
            print_info("  You'll need audio prompts to test inference")
    else:
        print_warning("prompts/ directory not found")
        print_info("  Consider creating it and adding test audio files")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_ok = script_ok and model_ok and deps_ok and checkpoint_ok
    
    if all_ok:
        print_success("All checks passed! ✨")
        print("\nYou can now run inference:")
        print(f"\n  {BLUE}./bin/generate_audio.sh \\")
        print(f"      -c YOUR_CHECKPOINT.pt \\")
        print(f"      -p YOUR_PROMPT.wav{RESET}")
        print("\nFor more examples, see AUDIO_TO_AUDIO_INFERENCE.md")
        return 0
    else:
        print_error("Some checks failed!")
        print("\nPlease fix the issues above before running inference.")
        print("See AUDIO_TO_AUDIO_INFERENCE.md for setup instructions.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
