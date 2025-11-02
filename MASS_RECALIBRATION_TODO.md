# Mass Recalibration - Future Port

## Priority: High

Mass recalibration improves mass accuracy systematically across a dataset, which is essential for high-confidence identifications.

## Key Test Requirement

**CRITICAL**: When implementing mass recalibration tests, include a test that explicitly checks **2+ vs 3+ charge state handling**.

### Why This Matters

Mass errors can be charge-state dependent due to:
1. Isotope selection differences (choosing wrong isotope peak)
2. Ion suppression effects
3. Space-charge effects in the ion trap

### Required Test

```python
def test_charge_state_specific_recalibration():
    """Test that mass recalibration handles 2+ and 3+ separately."""
    # Create PSMs with known mass errors for different charge states
    # 2+ peptides: systematic +5 ppm error
    # 3+ peptides: systematic -3 ppm error

    # After recalibration:
    # - 2+ peptides should have errors centered at 0 ppm
    # - 3+ peptides should have errors centered at 0 ppm
    # - Verify separate calibration curves were applied
```

### Implementation Notes

- Check if AlphaMod's `mass_recalibration.py` already handles this
- If not, implement charge-state-specific calibration curves
- Use robust regression (e.g., RANSAC or Huber loss) to handle outliers
- Consider RT-dependent recalibration as well (mass error can drift over gradient)

## Source Module

`alphamod/scoring/mass_recalibration.py` (9.3K, ~300 lines)

## Estimated Scope

- Port module: ~300 lines → ~400 lines
- Tests: ~20-25 tests including charge state test
- Effort: Low-Medium
- Value: HIGH - improves identification accuracy significantly
