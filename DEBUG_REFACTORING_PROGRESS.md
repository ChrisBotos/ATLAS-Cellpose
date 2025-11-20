# Debug Mode and Rich Console Integration - Progress Report

## OBJECTIVE
Systematically refactor debug and logging output across all scripts in `/code` directory to:
1. Respect `debug_mode` parameter from config file
2. Use Rich library for beautiful formatted output
3. Show minimal output when debug_mode=False (only critical milestones)
4. Show detailed output when debug_mode=True

## ✅ COMPLETED FILES

### 1. code/nuclei_segmentation/utils/logging_utils.py
- ✓ Added Rich console and RichHandler imports
- ✓ Replaced standard logging console handler with RichHandler
- ✓ Added Rich formatting to startup messages
- ✓ Updated setup_debug() to use console.print() for debug messages
- ✓ Console shows INFO and above, DEBUG only goes to log file

### 2. code/nuclei_segmentation/pipeline.py
- ✓ Added Rich console import
- ✓ Updated setup_model() to accept debug_mode parameter
- ✓ Replaced logger.info() with console.print() for major milestones
- ✓ Added color-coded success/error messages ([green]✓, [red]✗, [yellow]⚠)
- ✓ Added formatted numerical output with Rich styling
- ✓ Ensured debug_mode is extracted and passed through pipeline
- ✓ Updated run_segmentation_pipeline() to use Rich console
- ✓ Made many INFO messages conditional on debug_mode

### 3. code/nuclei_segmentation/run_this.py
- ✓ Added Rich console import
- ✓ Updated fatal error handling to use Rich formatting
- ✓ Improved traceback display with Rich

### 4. code/nuclei_segmentation/utils/segmentation.py (PARTIAL)
- ✓ Added Rich console and Progress imports
- ⚠ Still needs: 88 logger calls to be reviewed and updated

## 🔄 REMAINING WORK

### High Priority - Core Segmentation Files (6 files)
1. **segmentation.py** - 88 logger calls, needs careful review
2. **preprocessing.py** - Review all print/logger statements
3. **filter_masks.py** - Make verbose output conditional
4. **visualization.py** - Update overlay generation messages
5. **overlay_masks.py** - Update overlay messages
6. **parallel_segmentation.py** - Review parallel processing logs

### Medium Priority - Merge System (5 files)
7. **cellpose_merge/merge_tiles.py** - Already has debug_mode parameter
8. **cellpose_merge/two_phase_merge.py** - Review merge logging
9. **cellpose_merge/qc.py** - Review QC overlay messages
10. **cellpose_merge/cpu_merge.py** - Review merge messages
11. **cellpose_merge/gpu_merge.py** - Review merge messages

### Lower Priority - Other Utilities (4 files)
12. **utils/tiling.py** - Review tiling messages
13. **utils/watershed.py** - Review watershed messages
14. **utils/project_setup.py** - Review setup messages
15. **utils/debug_utils.py** - May need updates

### Feature Extraction (Already has Rich)
- engineered_feature_extraction/* - Already uses Rich extensively
- May need debug_mode parameter integration if not already present

## 📋 REFACTORING STANDARDS

### Debug Mode Behavior
- **debug_mode=False**: Only show critical pipeline milestones and final results
- **debug_mode=True**: Show detailed progress and diagnostic information
- **DEBUG-level logs**: Always go to file only, never to console

### Rich Formatting Standards
```python
[cyan]           # Section headers and info
[green]✓         # Success messages
[red]✗           # Errors
[yellow]⚠        # Warnings
[blue]ℹ          # Informational notes
[36m[1m{n}[/1m[/36m]  # Highlighted numbers
[dim]            # Less important details (debug mode only)
[1m]             # Bold for emphasis
```

### Console vs Logger Pattern
```python
# For important events - use both
console.print(f"[green]✓[/green] Segmentation completed: [36m[1m]{count:,}[/1m[/36m] nuclei")
logger.info(f"Segmentation completed: {count} nuclei")

# For debug info - logger only
if debug_mode:
    logger.debug(f"Tile parameters: {params}")
```

### Progress Bars
Use Rich Progress() for long-running operations:
```python
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

with Progress() as progress:
    task = progress.add_task("[cyan]Processing tiles...", total=n_tiles)
    for tile in tiles:
        # process tile
        progress.update(task, advance=1)
```

## 🎯 RECOMMENDED NEXT STEPS

1. **Complete segmentation.py** (highest impact - 88 logger calls)
2. **Update preprocessing.py** (user sees this early in pipeline)
3. **Update filter_masks.py** (important results filtering)
4. **Update merge system files** (critical for tiled processing)
5. **Update remaining utility files**
6. **Test with debug_mode=False** (ensure clean output)
7. **Test with debug_mode=True** (ensure detailed output)

## 📝 NOTES

- Rich library is already in requirements.txt (version 13.6.0)
- All files should import: `from rich.console import Console`
- Initialize once per file: `console = Console()`
- Logger still writes to file for audit trail
- Console provides beautiful user-facing output

