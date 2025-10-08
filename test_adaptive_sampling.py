#!/usr/bin/env python3
"""
Test script for adaptive boundary sampling with spike interpolation.

This script validates the enhanced sampling implementation by:
1. Generating synthetic data with known spike events
2. Testing spike detection and interpolation
3. Comparing original vs enhanced sampling
4. Validating that sampling concentrates on spike regions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.surf_flux import synth_surface_flux
from src.spike_detection import detect_spike_events
from src.adaptive_boundary_sampling import (
    adaptive_boundary_sampling,
    create_interpolated_spike_samples,
    visualize_sampling_distribution
)
from src.boundary_sampling import sample_boundary_points


def test_interpolated_spike_samples():
    """Test 1: Verify interpolated sample generation"""
    print("\n" + "="*70)
    print("TEST 1: Interpolated Spike Sample Generation")
    print("="*70)

    device = torch.device('cpu')

    # Create simple test data
    t_test = torch.linspace(0, 100, 50, device=device).view(-1, 1)

    # Create a single spike event (indices 20-25)
    spike_events = [torch.arange(20, 26, device=device)]

    # Test interpolation
    sampling_info = create_interpolated_spike_samples(
        t_test,
        spike_events,
        interpolation_density=3,
        neighborhood_expansion=2,
        device=device
    )

    print(f"Original time points: {len(t_test)}")
    print(f"Spike event size: {len(spike_events[0])}")
    print(f"Expanded spike indices: {len(sampling_info['spike_indices'])}")
    print(f"Interpolated times created: {len(sampling_info['interpolated_times'])}")
    print(f"Original weights shape: {sampling_info['original_weights'].shape}")

    # Verify weights are higher for spike regions
    spike_idx = sampling_info['spike_indices']
    non_spike_mask = ~torch.isin(torch.arange(len(t_test), device=device), spike_idx)

    avg_spike_weight = sampling_info['original_weights'][spike_idx].mean().item()
    avg_non_spike_weight = sampling_info['original_weights'][non_spike_mask].mean().item()

    print(f"Average spike weight: {avg_spike_weight:.2f}")
    print(f"Average non-spike weight: {avg_non_spike_weight:.2f}")

    assert avg_spike_weight > avg_non_spike_weight, "Spike weights should be higher!"
    assert len(sampling_info['interpolated_times']) > 0, "Should generate interpolated points!"

    print("PASSED: Interpolated samples generated correctly")
    return True


def test_adaptive_sampling_distribution():
    """Test 2: Verify adaptive sampling focuses on spikes"""
    print("\n" + "="*70)
    print("TEST 2: Adaptive Sampling Distribution")
    print("="*70)

    device = torch.device('cpu')

    # Generate synthetic data with spikes
    t, q = synth_surface_flux(
        total_days=10,
        dt_minutes=30,
        storm_rate_per_day=0.3,
        seed=42
    )

    q_tensor = torch.tensor(q, device=device)
    t_tensor = torch.tensor(t, device=device).view(-1, 1)

    # Detect spikes
    spike_events, _ = detect_spike_events(
        q_tensor,
        threshold_method='std',
        threshold_value=1.0,
        expansion_window=5,
        merge_distance=3,
        min_event_size=2,
        device=device
    )

    print(f"Data points: {len(t)}")
    print(f"Spike events detected: {len(spike_events)}")

    if len(spike_events) == 0:
        print("WARNING: No spikes detected, skipping this test")
        return True

    # Get all spike indices
    all_spike_indices = torch.cat(spike_events).unique()
    print(f"Total spike indices: {len(all_spike_indices)}")

    # Sample with adaptive sampling
    batch_size = 200
    t_bc_adaptive = adaptive_boundary_sampling(
        t_tensor,
        spike_events,
        n_events=len(spike_events),
        batch_size_bc=batch_size,
        device=device,
        spike_ratio=0.7,
        interpolation_density=3,
        neighborhood_expansion=2,
        use_weighted_sampling=True
    )

    print(f"Sampled {len(t_bc_adaptive)} points")

    # Count how many samples are near spike regions
    spike_times = t_tensor[all_spike_indices].flatten()
    dt_mean = (t_tensor[1:] - t_tensor[:-1]).mean().item()
    tolerance = 2.0 * dt_mean

    n_near_spike = 0
    for sample in t_bc_adaptive.detach().flatten():
        min_dist = torch.abs(spike_times - sample).min().item()
        if min_dist <= tolerance:
            n_near_spike += 1

    spike_coverage = n_near_spike / len(t_bc_adaptive)
    print(f"Spike coverage: {spike_coverage*100:.1f}%")

    # With spike_ratio=0.7, we expect ~70% coverage (allowing some tolerance)
    assert spike_coverage > 0.5, f"Expected >50% spike coverage, got {spike_coverage*100:.1f}%"

    print("PASSED: Adaptive sampling focuses on spike regions")
    return True


def test_comparison_with_original():
    """Test 3: Compare enhanced vs original sampling"""
    print("\n" + "="*70)
    print("TEST 3: Enhanced vs Original Sampling Comparison")
    print("="*70)

    device = torch.device('cpu')

    # Generate data
    t, q = synth_surface_flux(
        total_days=15,
        dt_minutes=30,
        storm_rate_per_day=0.4,
        seed=16
    )

    q_tensor = torch.tensor(q, device=device)
    t_tensor = torch.tensor(t, device=device).view(-1, 1)

    # Detect spikes
    spike_events, _ = detect_spike_events(
        q_tensor,
        threshold_method='std',
        threshold_value=1.0,
        expansion_window=5,
        merge_distance=3,
        min_event_size=2,
        device=device
    )

    if len(spike_events) == 0:
        print("WARNING: No spikes detected, skipping this test")
        return True

    batch_size = 200
    n_events = min(9, len(spike_events))

    # Original sampling
    t_bc_original = sample_boundary_points(
        t_tensor,
        spike_events,
        n_events,
        batch_size,
        device
    )

    # Enhanced sampling
    t_bc_enhanced = adaptive_boundary_sampling(
        t_tensor,
        spike_events,
        n_events,
        batch_size,
        device=device,
        spike_ratio=0.7,
        interpolation_density=3,
        neighborhood_expansion=2,
        use_weighted_sampling=True
    )

    print(f"Original sampling: {len(t_bc_original)} samples")
    print(f"Enhanced sampling: {len(t_bc_enhanced)} samples")

    # Verify both produce correct batch sizes
    assert len(t_bc_original) == batch_size, "Original sampling batch size mismatch"
    assert len(t_bc_enhanced) == batch_size, "Enhanced sampling batch size mismatch"

    # Verify gradients are enabled
    assert t_bc_original.requires_grad, "Original samples should have gradients enabled"
    assert t_bc_enhanced.requires_grad, "Enhanced samples should have gradients enabled"

    print("PASSED: Both sampling methods produce correct output shape and gradients")
    return True


def test_edge_cases():
    """Test 4: Edge cases (no spikes, single spike, etc.)"""
    print("\n" + "="*70)
    print("TEST 4: Edge Case Handling")
    print("="*70)

    device = torch.device('cpu')

    # Test case 1: No spikes
    print("\nTest 4.1: No spike events")
    t_test = torch.linspace(0, 100, 50, device=device).view(-1, 1)
    spike_events_empty = []

    t_bc_no_spikes = adaptive_boundary_sampling(
        t_test,
        spike_events_empty,
        n_events=0,
        batch_size_bc=20,
        device=device
    )

    assert len(t_bc_no_spikes) == 20, "Should handle no spikes gracefully"
    print("PASSED: Handles no spike events")

    # Test case 2: Single point spike
    print("\nTest 4.2: Single-point spike")
    spike_events_single = [torch.tensor([25], device=device)]

    t_bc_single = adaptive_boundary_sampling(
        t_test,
        spike_events_single,
        n_events=1,
        batch_size_bc=20,
        device=device,
        interpolation_density=2,
        neighborhood_expansion=1
    )

    assert len(t_bc_single) == 20, "Should handle single-point spike"
    print("PASSED: Handles single-point spike")

    # Test case 3: Very small batch size
    print("\nTest 4.3: Small batch size")
    t_bc_small = adaptive_boundary_sampling(
        t_test,
        spike_events_single,
        n_events=1,
        batch_size_bc=5,
        device=device
    )

    assert len(t_bc_small) == 5, "Should handle small batch sizes"
    print("PASSED: Handles small batch size")

    return True


def run_all_tests():
    """Run all validation tests"""
    print("\n" + "#"*70)
    print("# ADAPTIVE BOUNDARY SAMPLING TEST SUITE")
    print("#"*70)

    tests = [
        ("Interpolated Sample Generation", test_interpolated_spike_samples),
        ("Adaptive Sampling Distribution", test_adaptive_sampling_distribution),
        ("Enhanced vs Original Comparison", test_comparison_with_original),
        ("Edge Case Handling", test_edge_cases),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASSED" if result else "FAILED"))
        except Exception as e:
            print(f"\nERROR in {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))

    # Print summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)
    for name, status in results:
        status_symbol = "✓" if status == "PASSED" else "✗"
        print(f"{status_symbol} {name}: {status}")

    all_passed = all(status == "PASSED" for _, status in results)

    if all_passed:
        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("SOME TESTS FAILED!")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
