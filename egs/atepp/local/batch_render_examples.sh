#!/bin/bash
#
# Example batch processing script for rendering MIDI files with different plugins
# 
# Usage: ./batch_render_examples.sh
#

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR/.."

echo "=========================================="
echo "Batch MIDI Rendering Examples"
echo "=========================================="
echo ""

# Make sure Reaper is running
echo "⚠️  Make sure Reaper is running before proceeding!"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Example 1: Render samples with default plugin
echo ""
echo "Example 1: Rendering samples folder..."
echo "------------------------------------------"
python reaper_midi_to_audio.py \
    -i "$PROJECT_DIR/samples" \
    -o "$PROJECT_DIR/rendered_samples" \
    -p "ReaSynth"

# Example 2: Render prompts with a specific plugin
echo ""
echo "Example 2: Rendering prompts folder..."
echo "------------------------------------------"
# Change this to your actual violin plugin name
VIOLIN_PLUGIN="VST3: Kontakt 7 (Native Instruments)"

python reaper_midi_to_audio.py \
    -i "$PROJECT_DIR/prompts" \
    -o "$PROJECT_DIR/rendered_prompts" \
    -p "$VIOLIN_PLUGIN"

# Example 3: Render with high quality settings
echo ""
echo "Example 3: High quality render (48kHz)..."
echo "------------------------------------------"
python reaper_midi_to_audio.py \
    -i "$PROJECT_DIR/samples" \
    -o "$PROJECT_DIR/rendered_samples_hq" \
    -p "$VIOLIN_PLUGIN" \
    -sr 48000

echo ""
echo "=========================================="
echo "Batch rendering complete!"
echo "=========================================="
echo ""
echo "Output locations:"
echo "  - $PROJECT_DIR/rendered_samples"
echo "  - $PROJECT_DIR/rendered_prompts"
echo "  - $PROJECT_DIR/rendered_samples_hq"
echo ""
