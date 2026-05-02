# CLAUDE.md – ATLAS-Cellpose

This file provides **mandatory rules** and **project context** for Claude Code when editing or generating code in ATLAS-Cellpose.

---

## **1. Project Overview**

**ATLAS-Cellpose** (Adaptive Tiled Local Analysis Segmentation) is a computational framework for large-scale nuclear segmentation in tissue sections using Cellpose3 with adaptive parameter optimization through intelligent tiling.

Core capabilities:

- **Adaptive tiled processing** for gigapixel images (memory-efficient, locally optimized).
- **4-step merging algorithm** for systematic tile boundary resolution.
- **CLAHE preprocessing** with configurable contrast enhancement.
- **Morphological filtering** for segmentation quality control.
- **GPU-accelerated segmentation** with automatic CPU fallback (merge step is CPU-only).

**Primary application:** Ischemia-reperfusion injury studies in kidney tissue (DAPI-stained sections).

**This is a standalone project.** It does not depend on or interact with other repositories.

---

## **2. Directory Structure**

```
ATLAS-Cellpose/
├── src/
│   └── atlas_cellpose/                # Primary package (pip install -e .)
│       ├── __init__.py                # Public API + __version__
│       ├── run_this.py                # Pipeline entry point (Python)
│       ├── pipeline.py                # Core segmentation workflow
│       ├── cellpose_merge/            # Tile merging module
│       │   ├── __init__.py            # Re-exports
│       │   ├── cpu_merge.py           # CPU-based merging
│       │   ├── gpu_merge.py           # GPU merging (planned, not yet implemented)
│       │   ├── merge_tiles.py         # Main merge orchestration
│       │   ├── two_phase_merge.py     # 4-step merging algorithm
│       │   ├── qc.py                  # Merge quality control
│       │   ├── merge_file_utils.py    # File I/O for merging
│       │   ├── merge_memory.py        # Memory management
│       │   └── merge_id_management.py # ID management during merging
│       └── utils/                     # Utility modules
│           ├── __init__.py
│           ├── binary_mask_visualization.py  # Binary mask generation
│           ├── debug_utils.py         # Debug utilities
│           ├── filter_masks.py        # Morphological filtering
│           ├── logging_utils.py       # Logging setup
│           ├── overlay_masks.py       # Visualization overlays
│           ├── parallel_segmentation.py  # Parallel processing
│           ├── preprocessing.py       # CLAHE, gamma correction
│           ├── project_setup.py       # Configuration loading
│           ├── segmentation.py        # Cellpose integration
│           ├── tiling.py              # Adaptive tiling strategy
│           ├── visualization.py       # Quality control visuals
│           └── watershed.py           # Edge refinement
├── tests/                             # Unit and integration tests
├── tasks_and_tools/                   # Shell wrappers
│   └── run_segmentation_instance.sh   # Main execution script
├── configs/                           # Configuration directory
│   └── nuclei_segmentation_config.ini # Main configuration file
├── data/                              # Input image data (gitignored contents)
├── results/                           # Output results (gitignored contents)
├── logs/                              # Log files (gitignored contents)
├── pyproject.toml                     # Package build configuration
├── LICENSE                            # MIT license
├── requirements.txt                   # Python dependencies
├── cellpose3_environment_recommended.yml  # Conda environment (Python 3.10)
├── cellpose4_environment.yml          # Alternative Cellpose 4 environment
├── .gitignore
├── README.md
└── CLAUDE.md                          # This file
```

---

## **3. Running the Pipeline**

### **Recommended: Shell Script Wrapper**
```bash
# Run with custom parameters (recommended).
./tasks_and_tools/run_segmentation_instance.sh image_path "data/your_image.tif" crop_box "0.38,0.42,0.32,0.36"

# Run with specific Cellpose parameters.
./tasks_and_tools/run_segmentation_instance.sh job_name test_run cellprob_threshold -14 flow_threshold 0.8
```

### **Alternative: Direct Python Execution**
```bash
conda activate venv310_cellpose3
python src/atlas_cellpose/run_this.py
```

### **Run Tests**
```bash
pytest tests/ -v
```

---

## **4. Environment Expectations**

Claude must always assume:

- **You are running inside WSL** (Linux environment) or Windows.
- The conda environment is `venv310_cellpose3` (Python 3.10, Cellpose 3.0.10, CUDA 11.8).
- Create it from `cellpose3_environment_recommended.yml` and install the package in editable mode:

```bash
conda env create -f cellpose3_environment_recommended.yml
conda activate venv310_cellpose3
pip install -e . --no-deps
```

- The conda environment must be activated before running Python commands:

```bash
conda activate venv310_cellpose3
```

**Large file rule:**
Many image files and results exceed standard token limits. Claude must **never** load large files (`.tif`, `.npy`, `.npz`) in full. Instead, use targeted inspection:

- `grep`, `rg`, `sed -n "<start>,<end>p"` for code files.
- `head`, `tail` for log files.
- Reading only relevant sections by line number or symbol search.
- The `results/` directory can be **massive** (10+ GB). NEVER attempt to list or read it in full.

---

## **5. Instructions for Claude**

### **5.1 General Behavior**

- Prefer **minimal, local, safe edits** that preserve existing structure.
- **Do not** attempt large-scale rewrites or architectural changes unless explicitly asked.
- Maintain strict separation of concerns:
  - *Pipeline logic* = `src/atlas_cellpose/pipeline.py`
  - *Utilities* = `src/atlas_cellpose/utils/` modules (one responsibility per file)
  - *Merge logic* = `src/atlas_cellpose/cellpose_merge/` modules
  - *Configuration* = `configs/` directory
- ATLAS-Cellpose is a **segmentation pipeline**. Do not add unrelated functionality.
- Never use em dashes. Use hyphens (-) instead in all text.

### **5.2 Project Goal: Clean-Up and Organization**

The current priority for this project is **clean-up, organization, and systematic mistake catching**. When working on this codebase:

1. **Identify and flag inconsistencies**: mismatched variable names, dead code paths, unused imports, commented-out code blocks, duplicated logic.
2. **Verify correctness**: check that parameter names in the config INI match what the code actually reads, check that default values in code match those documented in the README.
3. **Check for silent failures**: places where exceptions are caught and swallowed, where None is returned without warning, where fallback behavior differs from documented behavior.
4. **Ensure documentation accuracy**: verify that docstrings, README sections, and inline comments match actual code behavior.
5. **Flag potential bugs**: off-by-one errors in tiling/merging, integer overflow risks with large images, race conditions in parallel processing, incorrect coordinate transformations.

When you encounter an issue during any task, **report it clearly** even if it is outside the scope of the current request.

### **5.3 Coding Style**

#### **Comments & Documentation**
- All explanatory comments must be full sentences ending with a **full stop**.
- Function-level docstrings must be **Google-style**, including:

```python
"""Short summary.

Args:
    param_name (type): Description.
Returns:
    type: Description.
Raises:
    ErrorType: Description.
"""
```

- Include parameter types, return types, assumptions, and biological meaning where relevant.

#### **Titles / Subtitles**
- Titles must use: `"""Title"""`
- Subtitles must use: `'''Subtitle'''`
- No alternative formats.

#### **Code Quality**
- Prefer **small, testable functions** rather than long monolithic blocks.
- Maintain vectorized NumPy operations; avoid Python loops in tight image processing paths.
- Strive for optimized efficient code that matches the style of the rest of the file.

#### **Behavior Preservation**
- Never alter segmentation, merging, or filtering behavior silently.
- Any modifications to:
  - the 4-step merging algorithm
  - Cellpose parameter handling
  - morphological filtering thresholds or logic
  - CLAHE preprocessing
  - tiling overlap calculations
  must be explicitly documented.

### **5.4 Performance Expectations**

- Always preserve or improve computational complexity.
- Image processing must be memory-efficient. Use chunked/tiled processing for large arrays.
- When editing merge or segmentation code, verify that GPU and CPU code paths remain consistent.
- Never load full-resolution images into memory when a tile or chunk would suffice.

### **5.5 Error Handling & Logging**

- Use `rich` for:
  - progress bars
  - informative status messages
  - highlighted errors/warnings
- When raising errors, include:
  - the pipeline stage (preprocessing, segmentation, merging, filtering, visualization)
  - relevant file paths or tile indices
  - image dimensions and memory context when relevant
- Use the project's existing `logging_utils.py` for log setup. Do not create alternative logging mechanisms.

### **5.6 Testing**

For **any** nontrivial code change:

- Add or update tests under `tests/`.
- Tests must cover:
  - configuration loading and validation
  - tiling calculations (tile count, overlap regions, edge tiles)
  - merge algorithm correctness (boundary nuclei preservation, duplicate elimination)
  - morphological filter thresholds
  - coordinate transformations between tile-local and global coordinates
- Use deterministic test data (small synthetic images) rather than requiring real tissue images.
- Run tests using:

```bash
pytest tests/ -v
```

**After any change to the merge algorithm, segmentation parameters, or the pipeline call chain**, run a short smoke-test on the cropped example:

```bash
conda activate venv310_cellpose3
./tasks_and_tools/run_segmentation_instance.sh job_name smoke_test image_path "data/IRI_regist_cropped.tif" crop_image False use_tiling True
```

### **5.7 Configuration Consistency**

Whenever introducing or modifying a parameter:

1. Update `configs/nuclei_segmentation_config.ini` with the parameter and its comment.
2. Update `src/atlas_cellpose/utils/project_setup.py` to load the parameter.
3. Update `tasks_and_tools/run_segmentation_instance.sh` if the parameter should be overridable from the command line.
4. Update `README.md` if the parameter is user-facing.
5. Verify that the default value in the config file matches the fallback value in the code.

---

## **6. Author Header Block**

For every **Python file** that Claude creates or substantially modifies, include:

```python
"""
Author: Christos Botos.
Affiliation: Human Genetics Department, Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: <filename>.py.
Description:
    <Brief description of what this file does.>

Dependencies:
    - Python >= 3.10.
    - <list relevant dependencies>

Usage:
    <how to run this script>
"""
```

---

## **7. Git Policy**

- Only use git (add, commit, stash, tag, push, pull) when the user explicitly asks.
- Never include a `Co-Authored-By` line or any other indication that the commit was AI-assisted.
- Commit messages must be full descriptive sentences ending with a period.
- Never leave comments related to version changes.

---

## **8. Tips & Tricks**

### **Working with This Codebase**

- **Cellpose version matters:** This project is optimized for **Cellpose 3.0.10**. Cellpose 4.x has a different API and produces different results for nuclei. Do not upgrade Cellpose or change API calls without explicit approval.
- **Two execution methods exist:** `tasks_and_tools/run_segmentation_instance.sh` (recommended, creates temporary configs) and direct `python src/atlas_cellpose/run_this.py` (requires manual config editing). The shell script is the source of truth for how parameters are passed.
- **The 4-step merge algorithm is critical.** It is the core innovation of this project. Any changes to `two_phase_merge.py` must be carefully reviewed: Step 1 (priority selection), Step 2 (border deletion), Step 3 (cross-boundary preservation), Step 4 (cleanup). Incorrectly modifying merge logic can silently lose or duplicate nuclei.
- **Tile overlap calculations are subtle.** The overlap fraction (default 0.2) combined with tile side length determines the overlap region in pixels. Off-by-one errors here cause merge artifacts. Always verify with `qc_overlays = True`.
- **Results directory is massive** (10+ GB for full runs). NEVER attempt to read or list it in full.
- **Large source files exist.** `src/atlas_cellpose/pipeline.py` (~1236 lines), `src/atlas_cellpose/cellpose_merge/qc.py` (~2018 lines), `src/atlas_cellpose/utils/overlay_masks.py` (~1236 lines). Use targeted line ranges or grep for specific functions.
- **Config INI has multiple sections.** Parameters live in `[general]`, `[clahe]`, `[cellpose]`, `[tiling]`, `[filtering]`, etc. When adding a parameter, place it in the correct section.

### **Working with Claude Code Effectively**

- **NEVER read large files in full.** Many source files exceed token limits. Use targeted line ranges or grep for specific functions.
- **Use subagents for exploration.** Delegate broad file searches to Explore agents to keep your main context clean.
- **Verify after changes.** Run `pytest tests/ -v`. For pipeline changes, run a smoke test on the cropped example image.
- **Background long runs.** Full-image segmentation can take minutes to hours depending on image size and GPU availability. Use `run_in_background` and monitor logs.

### **Common Pitfalls**

- **Coordinate systems:** Tile coordinates are (row, col) in NumPy but (x, y) in some visualization code. Always verify which convention is being used.
- **Memory:** Full-resolution kidney images are ~700 MB as TIFF. Loading multiple copies crashes 16 GB systems. Always use chunked processing.
- **GPU/CPU consistency:** The merge module has both `cpu_merge.py` and `gpu_merge.py`. Any algorithm change must be applied to both.
- **Filter thresholds:** The config file and README may list different default values. The config file is the source of truth.

---

## **9. Large File and Token Management**

### **CRITICAL: Context Window Hygiene**
- **NEVER** read entire large data files (TIFF, NPY, NPZ, CSV, PNG) into context.
- **NEVER** dump raw result files or logs. Summarize findings instead.
- **NEVER** cat/print files larger than ~200 lines without using offset/limit.
- Use subagents for broad codebase exploration (10+ files).
- Prefer targeted file reads (specific line ranges) over full-file reads.

### **Working with Large Mathematical Scripts**
- When a script contains extensive math (derivations, equations, matrix operations): read it in sections using offset/limit rather than loading the entire file.
- Focus on the specific function or section being modified, not the whole file.

### **Working with Large Result Files**
- Read only the first/last few lines to verify format and content.
- Use grep/search to find specific values rather than reading entire files.
- Summarize results in markdown tables rather than reading raw output.

### **Recommended: Create a .claudeignore**
- Add a `.claudeignore` file at the project root to prevent accidental reads of large files.
- Include: `data/`, `results/`, `logs/`, `*.csv`, `*.tsv`, `*.h5`, `*.hdf5`, `*.h5ad`, `*.parquet`, `*.pkl`, `*.npy`, `*.npz`, `*.gz`, `*.tif`, `*.tiff`, `*.log`, `*.png`, `*.jpg`.

---

---

## Results Repository Scheme

All experiment outputs live in self-contained, date-prefixed run directories under `results/`.

### Directory layout

```
results/
└── <YYYY-MM-DD>_<run_name>/
    ├── config.json                  # Run-level parameter/config snapshot
    ├── <phase_a>/
    │   ├── data/                    # Numerical outputs (CSV, TSV, JSON)
    │   ├── figures/                 # Plots and visualizations (PNG, PDF)
    │   └── logs/                    # Phase-specific log files
    └── <phase_b>/
        ├── data/
        ├── figures/
        └── logs/
```

### Rules

1. Every script that produces outputs accepts `--name <run_name>` (default: `"default_run"`).
2. **Date prefix**: auto-prepended (`YYYY-MM-DD`) when the run directory is first created.
3. **Run reuse**: if `results/*_<run_name>/` already exists, new phases are added into it (no new directory).
4. **Phase replacement**: if `<phase>/` already exists within the run, it is deleted and recreated.
5. **Config snapshot**: a `config.json` is saved/updated in the run root with parameters, timestamp, and script invocation.
6. **Logs live inside results**: no separate top-level `logs/` directory. Each phase keeps its logs co-located. SLURM logs go to `<phase>/logs/slurm/`.

---

## Handling Outside Reviewer Feedback

Whenever you receive a message marked "Outside Reviewer Feedback":

1. Do not revise immediately. First, respond to every critique individually.
2. For each point, state one of: AGREE / DISAGREE / PARTIAL - followed by one sentence of reasoning.
3. Only after addressing all points, produce the revised plan or code.
4. If you disagree with a critique, explain why clearly. Do not silently ignore or capitulate without reason.
5. If a critique reveals something you missed, say so explicitly - do not retroactively act like it was always planned.
