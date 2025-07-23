# Cellpose3 vs Cellpose4 Compatibility Analysis

## Executive Summary

Your current codebase was designed specifically for **Cellpose4** and uses several Cellpose4-specific features that require modifications to work with **Cellpose3**. The main compatibility issues involve model initialization, parameter handling, and return value processing.

## Key Compatibility Issues

### 1. Model Initialization API Changes

**Current Code (Cellpose4):**
```python
from cellpose import models
model = models.CellposeModel(model_type='nuclei', gpu=True)
```

**Required for Cellpose3:**
```python
from cellpose import models
model = models.Cellpose(model_type='nuclei', gpu=True)
```

**Files Affected:**
- `code/nuclei_segmentation/pipeline.py` (line 89)
- `README.md` (line 1083)
- `install_pytorch_gpu.py` (line 133)
- `update_gpu_environment.bat` (line 71)

### 2. Auto-Detection Parameter Differences

**Current Code (Cellpose4):**
```python
# Uses diameter=0 or diameter=None for auto-detection
cellpose_params = {
    "diameter": 0,  # or None
    "resample": True,  # Deprecated in Cellpose4 v4.0.1+
    # ...
}
```

**Required for Cellpose3:**
```python
# Cellpose3 uses diameter=0 for auto-detection, resample is still required
cellpose_params = {
    "diameter": 0,  # Auto-detection
    "resample": True,  # Still required in Cellpose3
    # ...
}
```

### 3. Model.eval() Return Value Differences

**Cellpose4 Returns:**
```python
# Returns (masks, flows, styles, diameters) - 4 elements
masks, flows, styles, diameters = model.eval(image, **params)
```

**Cellpose3 Returns:**
```python
# Returns (masks, flows, styles) - 3 elements typically
masks, flows, styles = model.eval(image, **params)
# Diameter information may not be available in the same format
```

**Files Affected:**
- `code/nuclei_segmentation/utils/segmentation.py` (lines 131, 465)
- `code/nuclei_segmentation/utils/parallel_segmentation.py` (lines 211, 224-231)
- All test files expecting 4-element return tuples

### 4. Parameter Handling Differences

**Current Configuration Issues:**
- `diameter = None` in config may not work the same way in Cellpose3
- `resample = True` is still required in Cellpose3 (not deprecated)
- Some parameters may have different default values

## Required Code Modifications

### 1. Update Model Initialization

**File: `code/nuclei_segmentation/pipeline.py`**
```python
# Change line 89 from:
model = models.CellposeModel(model_type=model_type, gpu=(device == 'cuda'))

# To:
model = models.Cellpose(model_type=model_type, gpu=(device == 'cuda'))
```

### 2. Update Configuration Handling

**File: `code/nuclei_segmentation/utils/project_setup.py`**
```python
# Update diameter handling (line 140):
# Change from:
"diameter": None if config.get("cellpose", "diameter", fallback="None") == "None" else config.getint("cellpose", "diameter", fallback=0),

# To:
"diameter": 0 if config.get("cellpose", "diameter", fallback="0") in ["None", "0"] else config.getint("cellpose", "diameter", fallback=0),
```

### 3. Update Return Value Processing

**File: `code/nuclei_segmentation/utils/segmentation.py`**
```python
# Update diameter extraction logic (around lines 134-150):
def extract_diameter_info(cellpose_results, logger, context=""):
    """Extract diameter information from Cellpose results, handling version differences."""
    if len(cellpose_results) >= 4 and cellpose_results[3] is not None:
        # Cellpose4 format with diameter info
        detected_diameters = cellpose_results[3]
        if isinstance(detected_diameters, (list, np.ndarray)) and len(detected_diameters) > 0:
            avg_diameter = float(np.mean(detected_diameters))
            logger.info(f"{context}: Auto-detected diameter = {avg_diameter:.1f}px")
            return avg_diameter
    else:
        # Cellpose3 format - diameter info may not be available
        logger.info(f"{context}: Diameter information not available in Cellpose3 format")
        return None
    return None
```

### 4. Update Test Files

**All test files need updates to handle 3-element return tuples:**
```python
# Change from:
mock_model.eval.return_value = (mock_masks, mock_flows, mock_styles, mock_diameters)

# To:
mock_model.eval.return_value = (mock_masks, mock_flows, mock_styles)
```

## Configuration File Updates

### Update `configs/nuclei_segmentation_config.ini`

```ini
[cellpose]
; Use diameter=0 for Cellpose3 auto-detection (not None)
diameter = 0
; Resample is still required in Cellpose3 (not deprecated)
resample = True
; Other parameters remain the same
flow_threshold = 0.9
cellprob_threshold = -12
```

## Environment-Specific Considerations

### Cellpose3 Environment (`iri310_cellpose3`)
- Use `models.Cellpose()` for model initialization
- Keep `resample=True` parameter
- Handle 3-element return tuples from `model.eval()`
- Use `diameter=0` for auto-detection

### Cellpose4 Environment (`iri310`)
- Use `models.CellposeModel()` for model initialization
- `resample` parameter is deprecated but still works
- Handle 4-element return tuples from `model.eval()`
- Can use `diameter=None` or `diameter=0` for auto-detection

## Recommended Implementation Strategy

1. **Create version-agnostic wrapper functions** that detect the Cellpose version and adapt accordingly
2. **Implement feature detection** to determine available return values
3. **Use configuration flags** to specify which Cellpose version is being used
4. **Maintain separate configuration files** for each version if needed

## Testing Strategy

1. **Run existing tests** with Cellpose3 to identify specific failures
2. **Create version-specific test fixtures** for both Cellpose3 and Cellpose4
3. **Test auto-detection functionality** specifically with both versions
4. **Validate diameter detection** and parameter handling differences

## Next Steps

1. Create the `iri310_cellpose3` environment manually (instructions provided above)
2. Implement the code modifications listed above
3. Test the pipeline with both environments
4. Create version-agnostic wrapper functions for future compatibility
5. Update documentation to reflect version differences

This analysis provides a roadmap for making your codebase compatible with both Cellpose3 and Cellpose4.
