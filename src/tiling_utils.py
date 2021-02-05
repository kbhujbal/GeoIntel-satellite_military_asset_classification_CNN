"""
GeoIntel Tiling Utilities
=========================

Helper functions for slicing large satellite images into manageable tiles.
This module provides manual tiling capabilities when not using SAHI's auto-slicer.

The Small Object Problem:
- Satellite images: 4000x4000 pixels
- Military vehicles: ~20x20 pixels
- Standard YOLO resize (640x640) destroys small object features
- Solution: Process overlapping 512x512 tiles

Usage:
    from src.tiling_utils import ImageTiler

    tiler = ImageTiler(tile_size=512, overlap_ratio=0.2)
    tiles = tiler.slice_image("large_satellite.jpg")
    # Process each tile...
    final_detections = tiler.merge_detections(tile_detections)
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Generator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Tile:
    """Represents a single tile from a sliced image."""
    image: np.ndarray
    x_offset: int  # Pixel offset from original image left
    y_offset: int  # Pixel offset from original image top
    width: int
    height: int
    tile_id: int
    row: int
    col: int

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Return (x_min, y_min, x_max, y_max) in original image coordinates."""
        return (
            self.x_offset,
            self.y_offset,
            self.x_offset + self.width,
            self.y_offset + self.height
        )


@dataclass
class Detection:
    """Represents a single detection in original image coordinates."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x_min, y_min, x_max, y_max
    tile_id: Optional[int] = None

    @property
    def center(self) -> Tuple[float, float]:
        """Return center point of detection."""
        x_min, y_min, x_max, y_max = self.bbox
        return ((x_min + x_max) / 2, (y_min + y_max) / 2)

    @property
    def area(self) -> float:
        """Return area of bounding box."""
        x_min, y_min, x_max, y_max = self.bbox
        return (x_max - x_min) * (y_max - y_min)


class ImageTiler:
    """
    Handles slicing large satellite images into overlapping tiles
    and merging detections back to original coordinates.
    """

    def __init__(
        self,
        tile_size: int = 512,
        overlap_ratio: float = 0.2,
        min_area_ratio: float = 0.1
    ):
        """
        Initialize the image tiler.

        Args:
            tile_size: Size of square tiles (pixels)
            overlap_ratio: Overlap between adjacent tiles (0.0 to 0.5)
            min_area_ratio: Minimum visible area ratio for edge objects
        """
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.min_area_ratio = min_area_ratio
        self.overlap_pixels = int(tile_size * overlap_ratio)
        self.stride = tile_size - self.overlap_pixels

    def calculate_tile_grid(
        self,
        image_width: int,
        image_height: int
    ) -> Tuple[int, int, List[Tuple[int, int]]]:
        """
        Calculate the tile grid for a given image size.

        Args:
            image_width: Width of the original image
            image_height: Height of the original image

        Returns:
            Tuple of (n_cols, n_rows, list of (x_offset, y_offset) for each tile)
        """
        # Calculate number of tiles needed in each dimension
        n_cols = max(1, int(np.ceil((image_width - self.overlap_pixels) / self.stride)))
        n_rows = max(1, int(np.ceil((image_height - self.overlap_pixels) / self.stride)))

        tile_positions = []
        for row in range(n_rows):
            for col in range(n_cols):
                x_offset = col * self.stride
                y_offset = row * self.stride

                # Ensure tile doesn't exceed image boundaries
                x_offset = min(x_offset, max(0, image_width - self.tile_size))
                y_offset = min(y_offset, max(0, image_height - self.tile_size))

                tile_positions.append((x_offset, y_offset))

        return n_cols, n_rows, tile_positions

    def slice_image(
        self,
        image: np.ndarray,
        return_generator: bool = False
    ) -> List[Tile] | Generator[Tile, None, None]:
        """
        Slice a large image into overlapping tiles.

        Args:
            image: Input image as numpy array (H, W, C)
            return_generator: If True, return generator instead of list

        Returns:
            List or generator of Tile objects
        """
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        height, width = image.shape[:2]
        n_cols, n_rows, positions = self.calculate_tile_grid(width, height)

        logger.info(f"Slicing {width}x{height} image into {n_cols}x{n_rows} = {len(positions)} tiles")
        logger.info(f"Tile size: {self.tile_size}x{self.tile_size}, Overlap: {self.overlap_pixels}px")

        def generate_tiles():
            for tile_id, (x_offset, y_offset) in enumerate(positions):
                # Extract tile
                tile_img = image[
                    y_offset:y_offset + self.tile_size,
                    x_offset:x_offset + self.tile_size
                ]

                # Handle edge tiles that might be smaller
                actual_height, actual_width = tile_img.shape[:2]

                yield Tile(
                    image=tile_img,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    width=actual_width,
                    height=actual_height,
                    tile_id=tile_id,
                    row=tile_id // n_cols,
                    col=tile_id % n_cols
                )

        if return_generator:
            return generate_tiles()
        return list(generate_tiles())

    def slice_image_from_path(self, image_path: str) -> List[Tile]:
        """
        Slice an image from file path.

        Args:
            image_path: Path to the image file

        Returns:
            List of Tile objects
        """
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self.slice_image(image)

    def tile_to_original_coords(
        self,
        tile: Tile,
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[int, int, int, int]:
        """
        Convert bounding box from tile coordinates to original image coordinates.

        Args:
            tile: The tile containing the detection
            bbox: Bounding box in tile coordinates (x_min, y_min, x_max, y_max)

        Returns:
            Bounding box in original image coordinates
        """
        x_min, y_min, x_max, y_max = bbox
        return (
            int(x_min + tile.x_offset),
            int(y_min + tile.y_offset),
            int(x_max + tile.x_offset),
            int(y_max + tile.y_offset)
        )

    def merge_detections(
        self,
        tile_detections: List[Tuple[Tile, List[Detection]]],
        iou_threshold: float = 0.5,
        method: str = "nms"
    ) -> List[Detection]:
        """
        Merge detections from all tiles, handling overlapping regions.

        Args:
            tile_detections: List of (Tile, detections) tuples
            iou_threshold: IOU threshold for merging duplicate detections
            method: Merging method - "nms" (Non-Max Suppression) or "soft_nms"

        Returns:
            List of merged detections in original image coordinates
        """
        all_detections = []

        # Convert all detections to original coordinates
        for tile, detections in tile_detections:
            for det in detections:
                # Convert bbox to original coordinates
                orig_bbox = self.tile_to_original_coords(tile, det.bbox)
                all_detections.append(Detection(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=orig_bbox,
                    tile_id=tile.tile_id
                ))

        if not all_detections:
            return []

        # Group by class for NMS
        detections_by_class: Dict[int, List[Detection]] = {}
        for det in all_detections:
            if det.class_id not in detections_by_class:
                detections_by_class[det.class_id] = []
            detections_by_class[det.class_id].append(det)

        # Apply NMS per class
        merged = []
        for class_id, class_dets in detections_by_class.items():
            if method == "nms":
                class_merged = self._apply_nms(class_dets, iou_threshold)
            elif method == "soft_nms":
                class_merged = self._apply_soft_nms(class_dets, iou_threshold)
            else:
                raise ValueError(f"Unknown merge method: {method}")
            merged.extend(class_merged)

        logger.info(f"Merged {len(all_detections)} tile detections into {len(merged)} final detections")
        return merged

    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Compute Intersection over Union between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _apply_nms(
        self,
        detections: List[Detection],
        iou_threshold: float
    ) -> List[Detection]:
        """Apply Non-Maximum Suppression to detections."""
        if not detections:
            return []

        # Sort by confidence (descending)
        sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)
        keep = []

        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)

            # Remove detections with high IOU
            sorted_dets = [
                det for det in sorted_dets
                if self._compute_iou(best.bbox, det.bbox) < iou_threshold
            ]

        return keep

    def _apply_soft_nms(
        self,
        detections: List[Detection],
        iou_threshold: float,
        sigma: float = 0.5,
        score_threshold: float = 0.01
    ) -> List[Detection]:
        """Apply Soft-NMS with Gaussian penalty."""
        if not detections:
            return []

        # Work with copies to avoid modifying originals
        dets = [Detection(
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=d.confidence,
            bbox=d.bbox,
            tile_id=d.tile_id
        ) for d in detections]

        keep = []

        while dets:
            # Find max confidence
            max_idx = max(range(len(dets)), key=lambda i: dets[i].confidence)
            best = dets.pop(max_idx)
            keep.append(best)

            # Apply Gaussian penalty to overlapping detections
            remaining = []
            for det in dets:
                iou = self._compute_iou(best.bbox, det.bbox)
                if iou > 0:
                    # Gaussian weight
                    weight = np.exp(-(iou ** 2) / sigma)
                    det.confidence *= weight

                if det.confidence >= score_threshold:
                    remaining.append(det)

            dets = remaining

        return keep


def visualize_tiling_grid(
    image_path: str,
    tile_size: int = 512,
    overlap_ratio: float = 0.2,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize the tiling grid on an image.

    Args:
        image_path: Path to the input image
        tile_size: Size of tiles
        overlap_ratio: Overlap between tiles
        output_path: Optional path to save visualization

    Returns:
        Image with tiling grid drawn
    """
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    tiler = ImageTiler(tile_size=tile_size, overlap_ratio=overlap_ratio)
    height, width = image.shape[:2]
    n_cols, n_rows, positions = tiler.calculate_tile_grid(width, height)

    # Draw grid
    vis_image = image.copy()
    for x_offset, y_offset in positions:
        # Draw tile rectangle
        cv2.rectangle(
            vis_image,
            (x_offset, y_offset),
            (x_offset + tile_size, y_offset + tile_size),
            (0, 255, 0),
            2
        )

    # Add info text
    info_text = f"Tiles: {n_cols}x{n_rows}={len(positions)} | Size: {tile_size} | Overlap: {int(overlap_ratio*100)}%"
    cv2.putText(vis_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(str(output_path), vis_image)
        logger.info(f"Saved tiling visualization to {output_path}")

    return vis_image


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Test tiling utilities")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--output", help="Output path for visualization")

    args = parser.parse_args()

    vis = visualize_tiling_grid(
        args.image,
        tile_size=args.tile_size,
        overlap_ratio=args.overlap,
        output_path=args.output
    )
    print(f"Visualization shape: {vis.shape}")
