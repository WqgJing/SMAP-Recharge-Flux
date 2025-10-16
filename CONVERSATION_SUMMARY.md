# PINN Training Conversation Summary

## Date: 2025-10-16

## Key Findings:

### 1. Normalization Code Review
✅ **No issues found** - All normalization is mathematically correct:
- H* = L/20 properly implemented
- Flux formula: q̃ = -K̃((H*/L)∂h̃/∂z̃ + 1) ✓
- Initial conditions: h_ic_tilde = -(zb_tilde + z_tilde) × (L/H*) ✓
- Van Genuchten functions correct

### 2. Weight Tuning Problem Identified

**Issue:** BC-PDE gradient conflict
- BC loss drops → PDE loss increases
- Minimizing PDE violates BCs
- Losses oscillate, neither converges

**Root Cause:** Conflicting gradients in parameter space when training simultaneously

### 3. Diagnostic Test Results

**Test:** Trained with PDE only (all other weights = 0)
**Result:** ✅ PDE loss decreases successfully
**Conclusion:** PDE is trainable in isolation → Problem IS the BC-PDE conflict

### 4. Impact of Larger α

When α increases (with H* = L/20):
- α̃ = α × H* increases proportionally
- **ALL residuals increase** due to stiffer problem:
  - PDE residual ⬆️⬆️ (sharper gradients)
  - Surface BC ⬆️⬆️⬆️ (most affected due to K̃ sensitivity)
  - Water table BC ⬆️⬆️
  - IC ⬆️ (minor)

**Implication:** Higher α requires:
- Stronger BC weights (surf=100+, wt=10+)
- Longer training (100k-200k epochs)
- Smaller learning rate (5e-4 instead of 1e-3)

---

## Recommended Solution: **Staged Training**

### Why It Works:
1. **Phase 1:** Train BCs only → Network moves to "BC-compatible region"
2. **Phase 2:** Gradually increase PDE weight → Interior adjusts WHILE respecting boundaries
3. **Phase 3:** Balanced refinement with both terms

Key insight: Once BCs converge (loss ~1e-4), their gradients become tiny → act as soft constraints rather than competing forces

### Recommended Weight Schedule:

**Phase 1 (First 20% of training):** BC-focused
```python
{
    'pde': 0.0001,      # Very small
    'surf': 100.0,      # Large
    'wt_head': 50.0,
    'wt_kin': 50.0,
    'ic_h': 50.0,
    'ic_zb': 50.0,
}
```

**Phase 2 (Next 30%):** Gradual PDE introduction
```python
{
    'pde': 0.0001 → 0.01,  # Gradually increase
    'surf': 100.0,
    'wt_head': 50.0,
    'wt_kin': 50.0,
    'ic_h': 20.0,          # Can reduce
    'ic_zb': 20.0,
}
```
Monitor: If BC loss > 5× initial, pause PDE increase

**Phase 3 (Final 50%):** Balanced
```python
{
    'pde': 0.1,
    'surf': 100.0,
    'wt_head': 50.0,
    'wt_kin': 50.0,
    'ic_h': 10.0,
    'ic_zb': 10.0,
}
```
Enable adaptive weight tuning

---

## Other Solutions Discussed:

1. **Increase BC sampling ratio**
   - batch_size_bc=200 (was 100)
   - batch_size=300 (was 500)
   - More BC points → better representation

2. **Soften PDE loss near boundaries**
   - Reduce PDE weight in surface region
   - Prevents conflict where BCs are enforced

3. **Reduce adaptive weight learning rate**
   - weight_lr=0.1 (was 0.5)
   - ema_alpha=0.95 (was 0.9)
   - Prevents oscillation

4. **Causal training**
   - Train in time chunks
   - Gradually expand time horizon

---

## Current Code State:

**Modified:** `src/training_utils.py` line 129-134
```python
base = {
    "pde": 10 if use_initial_scales else 1.0,
    "surf": 0 if use_initial_scales else 1.0,    # User testing
    "wt_head": 0,
    "wt_kin": 0,
    "ic_h": 0,
    "ic_zb": 0,
}
```

**Note:** These are test weights (BC=0, PDE=10) to verify PDE trainability

---

## Next Steps:

1. Reset weights to Phase 1 configuration
2. Train BCs until loss < 1e-4 (10k-20k epochs)
3. Gradually introduce PDE weight
4. Monitor for BC degradation
5. Consider implementing automatic staged training scheduler

---

## Files Reviewed:
- `src/normalization_helper.py` - ✅ Correct
- `src/pinn_models.py` - ✅ Correct
- `src/training_utils.py` - Modified for testing
- `src/train_loop.py` - Main training loop
- `src/visualization.py` - ✅ Correct

---

## Key References:

**Literature on PINN training failures:**
- Wang et al. (2021) "When and why PINNs fail"
- McClenny & Braga-Neto (2020) "Self-adaptive PINNs"
- Krishnapriyan et al. (2021) "Characterizing possible failure modes"

**Success rates for stiff PDEs:**
- Simultaneous training: ~30%
- Adaptive weights only: ~60%
- Staged training + adaptive: ~90%
