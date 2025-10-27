# Contributing to AlphaPeptFast

Thank you for your interest in contributing to AlphaPeptFast! This document outlines the standards and process for adding new functions to the library.

## Standards for Inclusion

### 1. Battle-Tested in Production

**Required:** Function must be used in 2+ production projects

Functions in AlphaPeptFast are extracted from real research code, not written speculatively. Before proposing a new function, ensure it has been:
- Used successfully in at least 2 different projects
- Validated on real data
- Proven to solve a recurring problem

### 2. Performance Requirements

**Required:** >100,000 operations/second

All functions must meet strict performance standards:
- Use vectorized NumPy operations (no Python loops over data)
- Numba @njit compilation for complex inner loops
- Binary search (O(log n)) on sorted arrays when applicable
- Two-pass construction (count → allocate → fill) for array building

**Performance Testing:**
```python
import time
import numpy as np

# Example: test with realistic data size
data = np.random.randn(100000)
start = time.time()
for _ in range(100):
    result = your_function(data)
elapsed = time.time() - start
ops_per_sec = (100 * len(data)) / elapsed
assert ops_per_sec > 100000, f"Too slow: {ops_per_sec:.0f} ops/sec"
```

### 3. Comprehensive Testing

**Required:** Tests with both toy data and real data validation

Every function must have:

**Toy data tests** (10-100 items):
```python
def test_function_basic():
    """Test with hand-crafted minimal data."""
    mz = np.array([100.0, 200.0, 300.0])
    result = your_function(mz, tolerance=20.0)
    assert len(result) == expected_length
    assert result[0] == expected_value
```

**Real data validation tests:**
```python
def test_function_validation():
    """Validate against known-good output from source project."""
    # Load real data from source project
    mz, intensity = load_test_data("alphamod_sample.npy")

    # Compare to reference implementation output
    result = your_function(mz, intensity)
    reference = load_reference("alphamod_reference_output.npy")

    np.testing.assert_allclose(result, reference, rtol=1e-6)
```

**Edge case tests:**
```python
def test_function_edge_cases():
    """Test boundary conditions."""
    # Empty array
    assert len(your_function(np.array([]))) == 0

    # Single element
    result = your_function(np.array([100.0]))
    assert result is not None

    # Extreme values
    result = your_function(np.array([1e-10, 1e10]))
    assert np.all(np.isfinite(result))
```

### 4. NumPy-Style Docstrings

**Required:** Complete documentation

```python
def your_function(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    ppm_tolerance: float = 20.0,
) -> np.ndarray:
    """Brief one-line description.

    Longer description explaining what the function does, why it's useful,
    and any important algorithmic details.

    Parameters
    ----------
    mz_array : np.ndarray
        Observed m/z values (sorted!)
    intensity_array : np.ndarray
        Corresponding peak intensities
    ppm_tolerance : float, optional
        Mass tolerance in parts per million (default: 20.0)

    Returns
    -------
    np.ndarray
        Description of return value

    Notes
    -----
    - RT values must be in seconds (not minutes)
    - Input mz_array must be sorted for binary search
    - Typical performance: >100k operations/sec

    Examples
    --------
    >>> mz = np.array([100.0, 200.0, 300.0])
    >>> result = your_function(mz, intensity, ppm_tolerance=10.0)
    >>> len(result)
    3

    References
    ----------
    .. [1] Original implementation: AlphaMod/src/core/example.py
    """
    # Implementation here
```

### 5. Handbook Compliance

**Required:** Follow all patterns from Computational Proteomics Handbook

- **RT units:** Always in seconds internally, minutes for display only
- **Mass tolerance:** Always PPM, never Da
- **Charge relationship:** Higher charge = LOWER m/z
- **Vectorization:** No Python loops over data
- **Binary search:** On sorted arrays
- **Validation:** Sanity checks against common sense

## Contribution Process

### 1. Identify Candidate Function

Check if function meets inclusion criteria:
- [ ] Used in 2+ production projects
- [ ] Proven to solve recurring problem
- [ ] Not already available in AlphaBase/AlphaTools

### 2. Extract and Refactor

**Extract cleanest implementation** (usually most recent project):
```bash
# Copy from source project
cp ~/projects/MSC_MS1_high_res/src/rt_calibration/spline.py \
   alphapeptfast/rt/calibration.py
```

**Refactor for generality:**
- Remove project-specific imports
- Remove hardcoded paths/filenames
- Make all parameters explicit (no globals)
- Add type hints
- Improve variable names for clarity

### 3. Add Comprehensive Tests

Create test file in `tests/`:
```python
# tests/test_rt_calibration.py
import numpy as np
import pytest
from alphapeptfast.rt import SplineWithLinearExtrapolation

def test_basic_calibration():
    """Test with toy data."""
    # ... toy data test

def test_validation_alphamod():
    """Validate against AlphaMod reference."""
    # ... real data validation

def test_validation_msc():
    """Validate against MSC_MS1_high_res reference."""
    # ... real data validation

def test_edge_cases():
    """Test boundary conditions."""
    # ... edge cases
```

### 4. Document Thoroughly

Add NumPy-style docstring with:
- Brief description
- Detailed explanation
- All parameters documented
- Return value documented
- Notes section (RT units, performance, requirements)
- Examples section
- References to original implementation

### 5. Validate Performance

Run performance benchmark:
```bash
pytest tests/test_your_module.py -v --benchmark
```

Ensure >100k ops/sec on typical workstation.

### 6. Update Module __init__.py

Add to module's `__init__.py`:
```python
from .calibration import SplineWithLinearExtrapolation

__all__ = ["SplineWithLinearExtrapolation"]
```

### 7. Submit Pull Request

**PR Title:** `Add [function_name] from [source_projects]`

**PR Description:**
```markdown
## Summary
Extracts SplineWithLinearExtrapolation from MSC_MS1_high_res

## Source Projects
- MSC_MS1_high_res: /src/rt_calibration/spline.py
- Used in AlphaMod (reinvented)
- Used in AlphaModFS (reinvented)

## Performance
- Calibration fit: Once per run (~10-100ms)
- Prediction: >1M peptides/sec

## Tests
- ✅ Toy data (10 points)
- ✅ Real data validation (MSC_MS1_high_res)
- ✅ Edge cases (empty, single point, extrapolation)
- ✅ Coverage: 100%

## Breaking Changes
None (new function)
```

## Development Setup

```bash
# Clone repository
git clone https://github.com/mannlab/AlphaPeptFast.git
cd AlphaPeptFast

# Create environment
uv venv --python 3.11
source .venv/bin/activate

# Install in development mode
uv pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=alphapeptfast --cov-report=term-missing

# Format code
black alphapeptfast tests
ruff check alphapeptfast tests
```

## Code Style

- **Formatting:** Use `black` (88 character line length)
- **Linting:** Use `ruff` (configured in pyproject.toml)
- **Type hints:** Required for all public functions
- **Naming:** Follow PEP 8 (snake_case for functions/variables)

## Questions?

Open an issue or discussion on GitHub:
https://github.com/mannlab/AlphaPeptFast/issues

Thank you for contributing to AlphaPeptFast!
