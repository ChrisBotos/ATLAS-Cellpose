"""
Watershed-based Nuclei Splitting for Kidney I/R Injury Analysis.

This module provides specialized watershed algorithms for splitting merged nuclei
in segmentation masks. This is particularly important for kidney tissue analysis
after ischemia-reperfusion injury, where inflammatory infiltrates can create
densely packed nuclear clusters that basic segmentation algorithms may fail to
separate properly.

The watershed algorithm uses distance transforms to identify likely centers of
individual nuclei within merged objects, then separates them based on these centers.
This is crucial for accurate quantification of nuclear counts and morphology in
regions of high cellular density.

This module also includes edge detection refinement for improving segmentation
boundaries using Canny edge detection, which can help separate touching nuclei
in densely packed regions of injured kidney tissue.
"""

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops
import logging
import traceback

def apply_watershed_to_mask(masks, min_area=1000, footprint=(3, 3), logger=None):
    """
    Apply watershed algorithm to split merged nuclei in a segmentation mask.

    This function identifies large objects in the segmentation mask that are likely
    to be merged nuclei, applies the watershed algorithm to split them based on
    distance transforms, and returns a refined mask with separated nuclei.

    Args:
        masks: Input segmentation mask as a numpy array.
        min_area: Minimum area threshold to identify potentially fused objects.
        footprint: Size of the local maxima filter for finding nuclear centers.
        logger: Optional logger for progress information.

    Returns:
        numpy.ndarray: Refined segmentation with split objects and reassigned labels.
    """
    if logger:
        logger.info(f"Applying watershed splitting to objects larger than {min_area} pixels")
        logger.info(f"Using local maxima footprint of {footprint}")
        logger.info(f"Input mask shape: {masks.shape}, dtype: {masks.dtype}")
        logger.info(f"Mask statistics: min={masks.min()}, max={masks.max()}, unique labels={len(np.unique(masks))}")

    # Verify mask is valid for watershed processing
    if masks.size == 0:
        if logger:
            logger.error("Empty mask provided to watershed function")
        return np.zeros((1, 1), dtype=np.uint32)  # Return minimal empty mask

    if masks.max() == 0:
        if logger:
            logger.warning("Mask contains no objects (all zeros)")
        return masks  # Return original mask as there's nothing to process

    # Initialize an empty mask to store the refined segmentation results
    final_mask = np.zeros_like(masks, dtype=np.uint32)

    try:
        # Extract properties of all labeled regions in the input mask
        props = regionprops(masks)

        if logger:
            logger.info(f"Found {len(props)} objects in the mask")

        # Initialize label counter for the new mask
        current_label = 0

        # Track statistics for reporting
        small_objects = 0
        large_objects = 0
        split_objects = 0

        # Process each region in the original segmentation mask
        for prop in props:
            # Skip background (label 0) if it somehow got included in regionprops
            if prop.label == 0:
                if logger:
                    logger.debug("Skipping background label (0) in watershed processing")
                continue

            # Get region properties
            area = prop.area
            minr, minc, maxr, maxc = prop.bbox  # Bounding box coordinates

            # CASE 1: Small objects - keep as is (likely single nuclei)
            if area <= min_area:
                # Assign a new label to this region
                current_label += 1

                # Copy the region to the final mask with the new label
                final_mask[prop.coords[:, 0], prop.coords[:, 1]] = current_label
                small_objects += 1

            # CASE 2: Large objects - apply watershed to split potential merged nuclei
            else:
                large_objects += 1
                if logger and large_objects % 100 == 0:
                    logger.info(f"Processing large object {large_objects}")

                # Extract the binary mask for this region
                submask = prop.image.astype(bool)

                # Compute distance transform - each pixel value is the distance to the nearest background pixel
                # This creates a topographic surface where nuclei centers are peaks
                distance = ndi.distance_transform_edt(submask)

                # Find local maxima in the distance map - these are likely nuclei centers
                # The footprint parameter controls the minimum separation between peaks
                peaks = peak_local_max(distance, footprint=np.ones(footprint), labels=submask)

                # Create a marker image for watershed
                marker = np.zeros(distance.shape, dtype=bool)

                # Place markers at the detected peaks (nuclei centers)
                if peaks.size > 0:
                    marker[tuple(peaks.T)] = True

                    # Label the markers for watershed
                    markers, num_markers = ndi.label(marker)

                    # Apply watershed to split the merged nuclei
                    # The negative distance is used so that watershed finds boundaries at the lowest points
                    # Between peaks (likely the boundaries between touching nuclei)
                    local_labels = watershed(-distance, markers, mask=submask)

                    # Assign new labels to each watershed-separated region
                    for ul in np.unique(local_labels):
                        # Skip background (label 0)
                        if ul == 0:
                            continue

                        # Assign a new unique label
                        current_label += 1

                        # Create a mask for this specific sub-region
                        region_mask = local_labels == ul

                        # Place the sub-region in the final mask at the correct position
                        # The bounding box coordinates are used to position the region correctly
                        final_mask[minr:maxr, minc:maxc][region_mask] = current_label
                        split_objects += 1
                else:
                    # No peaks found, keep the object as is
                    current_label += 1
                    final_mask[prop.coords[:, 0], prop.coords[:, 1]] = current_label

        if logger:
            logger.info(f"Watershed splitting complete:")
            logger.info(f"  Small objects kept as-is: {small_objects}")
            logger.info(f"  Large objects processed: {large_objects}")
            logger.info(f"  Objects after splitting: {split_objects + small_objects}")
            logger.info(f"  Total objects in final mask: {current_label}")

        return final_mask

    except Exception as e:
        if logger:
            logger.error(f"Error in watershed splitting: {e}")
            logger.error(traceback.format_exc())
        # Return the original mask if there's an error
        return masks


def refine_segmentation_with_edges(image, masks, settings, logger):
    """
    Refine segmentation masks using Canny edge detection.

    This function improves segmentation accuracy by incorporating edge information
    from the original image. In kidney tissue, nuclei often have distinct boundaries
    that may not be perfectly captured by Cellpose. By detecting these edges and
    using them to refine the segmentation, we can achieve more accurate delineation
    of nuclear boundaries, especially in densely packed regions.

    Args:
        image: Original grayscale image.
        masks: Initial segmentation masks from Cellpose.
        settings: Dictionary containing edge detection parameters.
        logger: Logger for progress information.

    Returns:
        numpy.ndarray: Refined segmentation masks with improved boundaries.
    """
    logger.info("Applying edge detection based refinement to the segmentation mask")

    # Verify that image and masks have the same shape
    if image.shape != masks.shape:
        logger.error(f"Shape mismatch: image {image.shape} vs masks {masks.shape}")
        logger.warning("Cannot apply edge detection with mismatched shapes")

        # Find common region that can be used for both
        common_h = min(image.shape[0], masks.shape[0])
        common_w = min(image.shape[1], masks.shape[1])

        logger.info(f"Using common region of size {common_h}x{common_w}")

        # Crop both to common size
        image = image[:common_h, :common_w]
        masks = masks[:common_h, :common_w]

        logger.info(f"Cropped image to {image.shape} and masks to {masks.shape}")

    # Apply Canny edge detection
    edges = cv2.Canny(image,
                      threshold1=settings.get("canny_threshold1", 50),
                      threshold2=settings.get("canny_threshold2", 150))

    # Dilate edges to ensure they fully separate touching nuclei
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Create binary mask from segmentation
    binary_mask = (masks > 0).astype(np.uint8) * 255

    # Subtract edges from binary mask
    refined_mask = cv2.subtract(binary_mask, dilated_edges)

    # Connected components analysis to get new labels
    num_labels, refined_labels = cv2.connectedComponents(refined_mask)

    logger.info(f"Refined segmentation into {num_labels - 1} objects after edge detection")
    return refined_labels
