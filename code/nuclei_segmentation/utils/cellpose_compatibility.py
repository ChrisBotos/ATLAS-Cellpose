"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: cellpose_compatibility.py.
Description:
    Version-agnostic wrapper for Cellpose3 and Cellpose4 compatibility.
    Provides unified interface for model initialization, parameter handling,
    and result processing across different Cellpose versions.

Dependencies:
    • Python >= 3.10.
    • cellpose (version 3.x or 4.x).
    • numpy for array operations.

Usage:
    from cellpose_compatibility import CellposeWrapper
    
    wrapper = CellposeWrapper(model_type='nuclei', gpu=True)
    masks, flows, n_cells, diameter_info = wrapper.segment(
        image=image_array,
        diameter=0,  # Auto-detection
        flow_threshold=0.9,
        cellprob_threshold=-12
    )

Arguments:
    model_type: Cellpose model type ('nuclei', 'cyto', etc.).
    gpu: Whether to use GPU acceleration.
    
Inputs:
    image: 2D numpy array of grayscale image.
    segmentation parameters: diameter, thresholds, etc.
    
Outputs:
    masks: 2D numpy array of segmentation masks.
    flows: Flow field information from Cellpose.
    n_cells: Number of detected cells/nuclei.
    diameter_info: Detected diameter information (if available).
    
Key Features:
    • Automatic Cellpose version detection.
    • Unified API for both Cellpose3 and Cellpose4.
    • Consistent parameter handling across versions.
    • Robust error handling and logging.
    • Diameter information extraction when available.
    
Notes:
    • Automatically detects Cellpose version and adapts API calls.
    • Handles parameter differences between versions gracefully.
    • Provides consistent return format regardless of version.
    • Logs version-specific information for debugging.
"""

import traceback
import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
from pathlib import Path

try:
    from cellpose import models
    import cellpose
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    models = None
    cellpose = None


class CellposeWrapper:
    """
    Version-agnostic wrapper for Cellpose3 and Cellpose4 compatibility.
    
    This class provides a unified interface for using Cellpose across different
    versions, automatically detecting the version and adapting API calls accordingly.
    """
    
    def __init__(self, model_type: str = 'nuclei', gpu: bool = True, logger: Optional[logging.Logger] = None):
        """
        Initialize the Cellpose wrapper with version detection.
        
        Args:
            model_type: Cellpose model type ('nuclei', 'cyto', etc.).
            gpu: Whether to use GPU acceleration.
            logger: Logger instance for status reporting.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model_type = model_type
        self.gpu = gpu
        self.model = None
        self.cellpose_version = None
        self.is_cellpose4 = False
        
        if not CELLPOSE_AVAILABLE:
            raise ImportError("Cellpose is not available. Please install cellpose.")
        
        # Detect Cellpose version.
        self._detect_version()
        
        # Initialize model based on version.
        self._initialize_model()
    
    def _detect_version(self):
        """Detect the installed Cellpose version and set compatibility flags."""
        try:
            # Get version string.
            if hasattr(cellpose, '__version__'):
                version_str = cellpose.__version__
            else:
                # Fallback version detection.
                version_str = "unknown"
            
            self.cellpose_version = version_str
            
            # Determine if this is Cellpose4 based on version or API availability.
            try:
                # Check if CellposeModel class exists (Cellpose4 feature).
                if hasattr(models, 'CellposeModel'):
                    self.is_cellpose4 = True
                    self.logger.info(f"Detected Cellpose4 (version: {version_str})")
                else:
                    self.is_cellpose4 = False
                    self.logger.info(f"Detected Cellpose3 (version: {version_str})")
            except Exception as e:
                # Fallback to version string parsing.
                if version_str.startswith('4') or '4.' in version_str:
                    self.is_cellpose4 = True
                    self.logger.info(f"Detected Cellpose4 via version string (version: {version_str})")
                else:
                    self.is_cellpose4 = False
                    self.logger.info(f"Detected Cellpose3 via version string (version: {version_str})")
                    
        except Exception as e:
            self.logger.warning(f"Could not detect Cellpose version: {e}")
            # Default to Cellpose3 for safety.
            self.is_cellpose4 = False
            self.cellpose_version = "unknown"
    
    def _initialize_model(self):
        """Initialize the Cellpose model based on detected version."""
        try:
            if self.is_cellpose4:
                # Use Cellpose4 API.
                self.model = models.CellposeModel(model_type=self.model_type, gpu=self.gpu)
                self.logger.info(f"Initialized Cellpose4 model: {self.model_type}")
            else:
                # Use Cellpose3 API.
                self.model = models.Cellpose(model_type=self.model_type, gpu=self.gpu)
                self.logger.info(f"Initialized Cellpose3 model: {self.model_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Cellpose model: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise e
    
    def segment(
        self,
        image: np.ndarray,
        diameter: Union[int, float, None] = 0,
        flow_threshold: float = 0.9,
        cellprob_threshold: float = -12,
        resample: bool = True,
        batch_size: int = 8,
        augment: bool = False,
        do_3D: bool = False
    ) -> Tuple[np.ndarray, Any, int, Optional[float]]:
        """
        Perform segmentation with version-agnostic parameter handling.
        
        Args:
            image: 2D numpy array of grayscale image.
            diameter: Expected diameter of objects (0 or None for auto-detection).
            flow_threshold: Threshold for flow gradient magnitude.
            cellprob_threshold: Threshold for cell probability.
            resample: Whether to resample image (required for Cellpose3).
            batch_size: Batch size for processing.
            augment: Whether to use augmentation.
            do_3D: Whether to process as 3D.
            
        Returns:
            Tuple of (masks, flows, n_cells, diameter_info).
        """
        try:
            # Prepare parameters based on version.
            eval_params = self._prepare_parameters(
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                resample=resample,
                batch_size=batch_size,
                augment=augment,
                do_3D=do_3D
            )
            
            # Log parameters for transparency.
            self.logger.info(f"Cellpose parameters: {eval_params}")
            
            # Ensure image has channel dimension.
            if image.ndim == 2:
                image_with_channels = image[..., None]
            else:
                image_with_channels = image
            
            # Run segmentation.
            cellpose_results = self.model.eval(image_with_channels, **eval_params)
            
            # Process results based on version.
            masks, flows, n_cells, diameter_info = self._process_results(cellpose_results)
            
            self.logger.info(f"Segmentation completed: {n_cells} objects detected")
            if diameter_info is not None:
                self.logger.info(f"Auto-detected diameter: {diameter_info:.1f}px")
            
            return masks, flows, n_cells, diameter_info
            
        except Exception as e:
            self.logger.error(f"Segmentation failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise e
    
    def _prepare_parameters(self, **kwargs) -> Dict[str, Any]:
        """Prepare parameters based on Cellpose version."""
        params = {}
        
        # Handle diameter parameter.
        diameter = kwargs.get('diameter', 0)
        if self.is_cellpose4:
            # Cellpose4 can use None or 0 for auto-detection.
            params['diameter'] = diameter
        else:
            # Cellpose3 uses 0 for auto-detection.
            params['diameter'] = 0 if diameter is None else diameter
        
        # Handle resample parameter.
        if self.is_cellpose4:
            # Resample is deprecated in Cellpose4 v4.0.1+ but still works.
            if kwargs.get('resample', True):
                params['resample'] = True
        else:
            # Resample is still required in Cellpose3.
            params['resample'] = kwargs.get('resample', True)
        
        # Common parameters.
        params.update({
            'flow_threshold': kwargs.get('flow_threshold', 0.9),
            'cellprob_threshold': kwargs.get('cellprob_threshold', -12),
            'batch_size': kwargs.get('batch_size', 8),
            'augment': kwargs.get('augment', False),
            'do_3D': kwargs.get('do_3D', False)
        })
        
        return params
    
    def _process_results(self, cellpose_results) -> Tuple[np.ndarray, Any, int, Optional[float]]:
        """Process Cellpose results based on version."""
        try:
            # Extract basic results (available in both versions).
            masks = cellpose_results[0]
            flows = cellpose_results[1] if len(cellpose_results) > 1 else None
            
            # Count detected objects.
            n_cells = len(np.unique(masks)) - 1  # Subtract 1 for background.
            
            # Extract diameter information if available.
            diameter_info = None
            if len(cellpose_results) >= 4 and cellpose_results[3] is not None:
                # Cellpose4 format with diameter info.
                detected_diameters = cellpose_results[3]
                if isinstance(detected_diameters, (list, np.ndarray)) and len(detected_diameters) > 0:
                    diameter_info = float(np.mean(detected_diameters))
            
            return masks, flows, n_cells, diameter_info
            
        except Exception as e:
            self.logger.error(f"Failed to process Cellpose results: {e}")
            # Return basic results even if diameter extraction fails.
            masks = cellpose_results[0] if len(cellpose_results) > 0 else np.zeros((100, 100), dtype=np.uint16)
            flows = cellpose_results[1] if len(cellpose_results) > 1 else None
            n_cells = len(np.unique(masks)) - 1
            return masks, flows, n_cells, None
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get information about the detected Cellpose version."""
        return {
            'version': self.cellpose_version,
            'is_cellpose4': self.is_cellpose4,
            'model_type': self.model_type,
            'gpu_enabled': self.gpu,
            'model_initialized': self.model is not None
        }


def create_cellpose_wrapper(model_type: str = 'nuclei', gpu: bool = True, logger: Optional[logging.Logger] = None) -> CellposeWrapper:
    """
    Factory function to create a CellposeWrapper instance.
    
    Args:
        model_type: Cellpose model type.
        gpu: Whether to use GPU acceleration.
        logger: Logger instance.
        
    Returns:
        Initialized CellposeWrapper instance.
    """
    return CellposeWrapper(model_type=model_type, gpu=gpu, logger=logger)
