#!/bin/bash

# Aletheia Script Runner
# Usage: ./run.sh [command] [args...]

PYTHON="./venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: Virtual environment python not found at $PYTHON"
    echo "Please run: 'uv venv --python 3.12 && source .venv/bin/activate && pip install -e .'"
    exit 1
fi

show_help() {
    echo "Usage: ./run.sh [command] [args]"
    echo ""
    echo "Available commands:"
    echo "  benchmark           Run the full benchmark suite"
    echo "  semantic            Run semantic consistency benchmark"
    echo "  adversarial         Generate adversarial test data (REQ: GEMINI_API_KEY)"
    echo "  fetch [url]         Fetch URL and test detection"
    echo "  train               Train the ensemble model"
    echo "  train-enhanced      Train the enhanced model"
    echo "  gen-training        Generate synthetic training data"
    echo "  verify              Verify Gemini detection capabilities"
    echo "  tests               Run the test suite"
    echo "  help                Show this help message"
    echo ""
    echo "Example:"
    echo "  ./run.sh fetch https://example.com"
}

COMMAND=$1
shift # Shift arguments so $@ contains the rest

case "$COMMAND" in
    benchmark)
        echo "🚀 Running Benchmark Suite..."
        $PYTHON scripts/benchmark_suite.py "$@"
        ;;
    semantic)
        echo "🧠 Running Semantic Benchmark..."
        $PYTHON scripts/benchmark_semantic.py "$@"
        ;;
    adversarial)
        echo "😈 Generating Adversarial Data..."
        $PYTHON scripts/generate_adversarial_data.py "$@"
        ;;
    fetch)
        echo "🌐 Fetching and Testing..."
        $PYTHON scripts/fetch_and_test.py "$@"
        ;;
    train)
        echo "🏋️  Training Ensemble Model..."
        $PYTHON scripts/train_ensemble.py "$@"
        ;;
    train-enhanced)
        echo "💪 Training Enhanced Model..."
        $PYTHON scripts/train_enhanced_model.py "$@"
        ;;
    gen-training)
        echo "📝 Generating Training Data..."
        $PYTHON scripts/generate_training_data.py "$@"
        ;;
    verify)
        echo "🔍 Verifying Gemini Detection..."
        $PYTHON scripts/verify_gemini_detection.py "$@"
        ;;
    tests)
        echo "🧪 Running Tests..."
        ./run_test.sh
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
