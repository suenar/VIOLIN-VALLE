#!/bin/bash
# Simple wrapper script for audio-to-audio generation

set -e

# Default values
CHECKPOINT=""
PROMPT=""
OUTPUT_DIR="infer/audio_continuation"
OUTPUT_FILE="generated"
MAX_DURATION=10.0
TOP_K=-100
TEMPERATURE=1.0

# Help message
show_help() {
    cat << EOF
Audio-to-Audio Generation Script

Usage: $0 -c CHECKPOINT -p PROMPT [OPTIONS]

Required:
    -c, --checkpoint PATH      Path to model checkpoint
    -p, --prompt PATH          Path to audio prompt file

Optional:
    -o, --output-dir DIR       Output directory (default: infer/audio_continuation)
    -f, --output-file NAME     Output filename prefix (default: generated)
    -d, --duration SECONDS     Max duration to generate (default: 10.0)
    -k, --top-k VALUE          Top-k sampling (default: -100)
    -t, --temperature VALUE    Sampling temperature (default: 1.0)
    -h, --help                 Show this help message

Examples:
    # Basic usage
    $0 -c exp/VALLE-audio/checkpoint-20000.pt -p prompts/prompt_A.wav

    # With custom parameters
    $0 -c exp/VALLE-audio/checkpoint-20000.pt \\
       -p prompts/prompt_A.wav \\
       -d 15.0 -k 50 -t 0.8 \\
       -o results -f my_generation

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -d|--duration)
            MAX_DURATION="$2"
            shift 2
            ;;
        -k|--top-k)
            TOP_K="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$CHECKPOINT" ] || [ -z "$PROMPT" ]; then
    echo "Error: Missing required arguments"
    echo ""
    show_help
    exit 1
fi

# Check if files exist
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$PROMPT" ]; then
    echo "Error: Prompt file not found: $PROMPT"
    exit 1
fi

# Run inference
echo "=========================================="
echo "Audio-to-Audio Generation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Prompt: $PROMPT"
echo "Output dir: $OUTPUT_DIR"
echo "Max duration: ${MAX_DURATION}s"
echo "Top-k: $TOP_K"
echo "Temperature: $TEMPERATURE"
echo "=========================================="
echo ""

python3 bin/infer_audiolm.py \
    --checkpoint "$CHECKPOINT" \
    --audio-prompts "$PROMPT" \
    --output-dir "$OUTPUT_DIR" \
    --output-file "$OUTPUT_FILE" \
    --max-duration "$MAX_DURATION" \
    --top-k "$TOP_K" \
    --temperature "$TEMPERATURE"

echo ""
echo "=========================================="
echo "Generation complete!"
echo "Output files:"
echo "  - ${OUTPUT_DIR}/${OUTPUT_FILE}_0_full.wav"
echo "  - ${OUTPUT_DIR}/${OUTPUT_FILE}_0_continuation.wav"
echo "=========================================="
