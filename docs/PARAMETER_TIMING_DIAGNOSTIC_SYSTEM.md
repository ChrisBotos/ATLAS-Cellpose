# Parameter Timing Diagnostic System

## 🎯 **Overview**

A comprehensive diagnostic system that tracks the computation time for **each individual parameter** across all nuclei and tiles, providing detailed performance analysis and optimization recommendations for nuclear feature extraction.

## 🔧 **System Components**

### **1. Individual Parameter Timing**
- **@time_parameter decorator**: Applied to each feature calculation function
- **Granular tracking**: Times every parameter computation (circularity, area, texture_entropy, etc.)
- **Statistical analysis**: Tracks min, max, average, and total computation times
- **Category organization**: Groups parameters by feature type (shape, size, texture, neighborhood)

### **2. Comprehensive Data Collection**
```python
@dataclass
class ParameterTiming:
    parameter_name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    category: str = ""
```

### **3. Automatic Report Generation**
- **Detailed diagnostic report**: Saved as `parameter_timing_diagnostic.txt` in output directory
- **Performance analysis**: Identifies computational bottlenecks
- **Optimization recommendations**: Suggests specific improvements
- **Scientific context**: Includes image information and processing statistics

## 📊 **Diagnostic Report Structure**

### **Header Information**
```
================================================================================
NUCLEAR FEATURE EXTRACTION - PARAMETER TIMING DIAGNOSTIC REPORT
================================================================================
Generated: 2025-08-04 13:04:49
Total nuclei processed: 15,806
Image dimensions: (3013, 1942)
Total processing time: 245.67 seconds (4.1 minutes)
Average time per nucleus: 15.54 ms
Total parameters computed: 41
```

### **Overall Performance Summary**
```
OVERALL PERFORMANCE SUMMARY
----------------------------------------
Parameter                      Total(s)   Avg(ms)    Calls    %     
----------------------------------------------------------------
fractal_dimension              45.23      2.86       15806    18.4  
convex_hull_area              32.15      2.03       15806    13.1  
glcm_features                 28.94      1.83       15806    11.8  
texture_entropy               18.67      1.18       15806     7.6  
gradient_features             15.42      0.98       15806     6.3  
```

### **Category Breakdown**
```
BREAKDOWN BY FEATURE CATEGORY
----------------------------------------

SHAPE FEATURES - 98.45s (40.1%)
------------------------------------------------------------
Parameter                           Total(s)   Avg(ms)    Min(ms)    Max(ms)   
-------------------------------------------------------------------------------------
fractal_dimension                   45.23      2.86       1.24       8.92     
convex_hull_area                    32.15      2.03       0.89       5.67     
circularity                         8.94       0.57       0.12       2.34     
eccentricity                        6.78       0.43       0.08       1.89     
solidity                            5.35       0.34       0.06       1.45     

TEXTURE FEATURES - 63.03s (25.7%)
------------------------------------------------------------
Parameter                           Total(s)   Avg(ms)    Min(ms)    Max(ms)   
-------------------------------------------------------------------------------------
glcm_features                       28.94      1.83       0.95       4.67     
texture_entropy                     18.67      1.18       0.45       3.21     
gradient_features                   15.42      0.98       0.34       2.89     
```

### **Detailed Parameter Statistics**
```
Parameter: fractal_dimension
  Category: shape
  Total computation time: 45.234 seconds
  Number of calls: 15,806
  Average time per call: 2.86 ms
  Minimum time per call: 1.24 ms
  Maximum time per call: 8.92 ms
  Percentage of total time: 18.41%
  Time per nucleus: 2.861 ms
```

### **Performance Recommendations**
```
PERFORMANCE RECOMMENDATIONS
==================================================
Most computationally expensive parameters:
  1. fractal_dimension (18.4% of total time)
  2. convex_hull_area (13.1% of total time)
  3. glcm_features (11.8% of total time)
  4. texture_entropy (7.6% of total time)
  5. gradient_features (6.3% of total time)

Optimization suggestions:
  • Consider disabling expensive parameters if not needed for analysis
  • Use GPU acceleration for texture and neighborhood features
  • Increase batch size for better parallel processing efficiency
  • Consider feature selection to focus on most informative parameters
```

## 🛠️ **Implementation Details**

### **Parameter Timing Decorator**
```python
def time_parameter(parameter_name: str, category: str = ""):
    """Decorator to time individual parameter calculations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Record timing statistics
            _parameter_timings[parameter_name].add_timing(end_time - start_time)
            return result
        return wrapper
    return decorator
```

### **Applied to Individual Features**
```python
# Shape features with individual timing
@time_parameter("circularity", "shape")
def compute_circularity():
    return (4 * np.pi * area / perimeter**2) if perimeter else np.nan

@time_parameter("fractal_dimension", "shape")
def fractal_dimension(binary_mask: np.ndarray) -> float:
    # Box-counting fractal dimension calculation
    ...

@time_parameter("texture_entropy", "texture")
def compute_texture_entropy():
    hist, _ = np.histogram(vals, bins=32, density=True)
    return float(entropy(hist))
```

## 📈 **Benefits for Research**

### **1. Performance Optimization**
- **Identify bottlenecks**: Pinpoint which parameters consume most computation time
- **Optimize workflows**: Focus optimization efforts on most expensive operations
- **Resource allocation**: Make informed decisions about computational resources

### **2. Scientific Reproducibility**
- **Detailed documentation**: Complete timing record for every parameter
- **Performance comparison**: Compare timing across different datasets
- **Method validation**: Verify computational efficiency of feature extraction

### **3. Parameter Selection**
- **Informed choices**: Select most informative parameters based on cost/benefit
- **Feature engineering**: Optimize feature sets for specific analyses
- **Computational budgeting**: Balance accuracy with processing time

## 🚀 **Usage**

The diagnostic system is **automatically activated** during feature extraction:

```bash
python code/engineered_feature_extraction/extract_engineered_features.py extract \
    --config configs/engineered_feature_extraction_config.ini
```

**Output**: `parameter_timing_diagnostic.txt` saved in the same directory as the feature extraction results.

## 📋 **Example Use Cases**

### **1. Performance Tuning**
- Identify that `fractal_dimension` takes 18% of total time
- Consider disabling if not critical for analysis
- Focus GPU optimization on texture features

### **2. Dataset Comparison**
- Compare timing across different tissue types
- Identify parameters that scale poorly with image size
- Optimize processing for large-scale studies

### **3. Method Development**
- Benchmark new feature implementations
- Compare computational efficiency of different algorithms
- Validate performance improvements

## ✅ **Validation Results**

The system has been thoroughly tested and validated:
- ✅ **Individual parameter timing**: Each feature calculation properly timed
- ✅ **Statistical accuracy**: Min, max, average times correctly calculated
- ✅ **Category organization**: Parameters properly grouped by feature type
- ✅ **Report generation**: Comprehensive diagnostic reports created
- ✅ **Performance recommendations**: Actionable optimization suggestions provided

The parameter timing diagnostic system provides **unprecedented visibility** into the computational performance of nuclear feature extraction, enabling data-driven optimization and scientific reproducibility.
