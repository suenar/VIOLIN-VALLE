#!/usr/bin/env python3
"""
Test script for AudioLM model
Verifies basic functionality without requiring actual data
"""

import torch
from valle.models import AudioLM


def test_audio_lm_forward():
    """Test forward pass with dummy data"""
    print("=" * 50)
    print("Testing AudioLM Forward Pass")
    print("=" * 50)
    
    # Model parameters
    batch_size = 2
    prompt_len = 150  # 3 seconds at 50Hz
    target_len = 250  # 5 seconds at 50Hz
    num_quantizers = 4
    
    # Initialize model
    model = AudioLM(
        d_model=256,  # Small for testing
        nhead=4,
        num_layers=2,
        num_quantizers=num_quantizers,
        norm_first=True,
        add_prenet=False,
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy data
    x = torch.randint(0, 2048, (batch_size, prompt_len, num_quantizers))
    x_lens = torch.tensor([prompt_len, prompt_len])
    
    y = torch.randint(0, 2048, (batch_size, target_len, num_quantizers))
    y_lens = torch.tensor([target_len, target_len])
    
    print(f"\nInput shapes:")
    print(f"  Prompt (x): {x.shape}")
    print(f"  Prompt lens: {x_lens}")
    print(f"  Target (y): {y.shape}")
    print(f"  Target lens: {y_lens}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions, loss, metrics = model(
            x=x,
            x_lens=x_lens,
            y=y,
            y_lens=y_lens,
            train_stage=0,
        )
    
    print(f"\nForward pass successful!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Metrics: {metrics}")
    
    return True


def test_audio_lm_inference():
    """Test inference with dummy data"""
    print("\n" + "=" * 50)
    print("Testing AudioLM Inference")
    print("=" * 50)
    
    # Model parameters
    prompt_len = 150  # 3 seconds
    num_quantizers = 4
    
    # Initialize model
    model = AudioLM(
        d_model=256,
        nhead=4,
        num_layers=2,
        num_quantizers=num_quantizers,
        norm_first=True,
    )
    
    # Create dummy prompt
    x = torch.randint(0, 2048, (1, prompt_len, num_quantizers))
    x_lens = torch.tensor([prompt_len])
    
    print(f"\nPrompt shape: {x.shape}")
    
    # Generate
    model.eval()
    with torch.no_grad():
        generated = model.inference(
            x=x,
            x_lens=x_lens,
            max_new_tokens=100,  # Generate 100 tokens (~2 seconds)
            temperature=1.0,
            top_k=50,
        )
    
    print(f"Generated shape: {generated.shape}")
    print(f"Generated {generated.shape[1] - prompt_len} new tokens")
    
    assert generated.shape[0] == 1
    assert generated.shape[2] == num_quantizers
    assert generated.shape[1] > prompt_len
    
    print("\nInference successful!")
    
    return True


def test_staged_training():
    """Test stage-specific parameter selection"""
    print("\n" + "=" * 50)
    print("Testing Staged Training")
    print("=" * 50)
    
    model = AudioLM(
        d_model=256,
        nhead=4,
        num_layers=2,
        num_quantizers=4,
    )
    
    # Test stage 1 (AR only)
    ar_params = list(model.stage_parameters(stage=1))
    print(f"\nStage 1 (AR) parameters: {len(ar_params)}")
    
    # Test stage 2 (NAR only)
    nar_params = list(model.stage_parameters(stage=2))
    print(f"Stage 2 (NAR) parameters: {len(nar_params)}")
    
    assert len(ar_params) > 0
    assert len(nar_params) > 0
    
    print("\nStaged training parameter selection working!")
    
    return True


def test_different_lengths():
    """Test with variable sequence lengths"""
    print("\n" + "=" * 50)
    print("Testing Variable Sequence Lengths")
    print("=" * 50)
    
    model = AudioLM(
        d_model=256,
        nhead=4,
        num_layers=2,
        num_quantizers=4,
    )
    
    batch_size = 3
    max_prompt_len = 200
    max_target_len = 300
    
    # Create variable length sequences
    x = torch.randint(0, 2048, (batch_size, max_prompt_len, 4))
    x_lens = torch.tensor([150, 180, 200])
    
    y = torch.randint(0, 2048, (batch_size, max_target_len, 4))
    y_lens = torch.tensor([250, 280, 300])
    
    print(f"\nBatch with variable lengths:")
    print(f"  Prompt lens: {x_lens.tolist()}")
    print(f"  Target lens: {y_lens.tolist()}")
    
    model.eval()
    with torch.no_grad():
        predictions, loss, metrics = model(
            x=x,
            x_lens=x_lens,
            y=y,
            y_lens=y_lens,
            train_stage=0,
        )
    
    print(f"\nVariable length forward pass successful!")
    print(f"  Loss: {loss.item():.4f}")
    
    return True


def test_ar_only():
    """Test AR stage only"""
    print("\n" + "=" * 50)
    print("Testing AR Stage Only")
    print("=" * 50)
    
    model = AudioLM(
        d_model=256,
        nhead=4,
        num_layers=2,
        num_quantizers=4,
    )
    
    x = torch.randint(0, 2048, (2, 150, 4))
    x_lens = torch.tensor([150, 150])
    y = torch.randint(0, 2048, (2, 250, 4))
    y_lens = torch.tensor([250, 250])
    
    model.eval()
    with torch.no_grad():
        predictions, loss, metrics = model(
            x=x,
            x_lens=x_lens,
            y=y,
            y_lens=y_lens,
            train_stage=1,  # AR only
        )
    
    print(f"\nAR stage only:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Metrics: {list(metrics.keys())}")
    assert "ArTop10Accuracy" in metrics
    assert "NarTop10Accuracy" not in metrics
    
    print("AR stage only training working!")
    
    return True


def test_nar_only():
    """Test NAR stage only"""
    print("\n" + "=" * 50)
    print("Testing NAR Stage Only")
    print("=" * 50)
    
    model = AudioLM(
        d_model=256,
        nhead=4,
        num_layers=2,
        num_quantizers=4,
    )
    
    x = torch.randint(0, 2048, (2, 150, 4))
    x_lens = torch.tensor([150, 150])
    y = torch.randint(0, 2048, (2, 250, 4))
    y_lens = torch.tensor([250, 250])
    
    model.eval()
    with torch.no_grad():
        predictions, loss, metrics = model(
            x=x,
            x_lens=x_lens,
            y=y,
            y_lens=y_lens,
            train_stage=2,  # NAR only
        )
    
    print(f"\nNAR stage only:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Metrics: {list(metrics.keys())}")
    assert "NarTop10Accuracy" in metrics
    assert "ArTop10Accuracy" not in metrics
    
    print("NAR stage only training working!")
    
    return True


def main():
    """Run all tests"""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  AudioLM Model Test Suite".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    
    tests = [
        ("Forward Pass", test_audio_lm_forward),
        ("Inference", test_audio_lm_inference),
        ("Staged Training", test_staged_training),
        ("Variable Lengths", test_different_lengths),
        ("AR Only", test_ar_only),
        ("NAR Only", test_nar_only),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {name} test PASSED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed! AudioLM is ready to use.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
