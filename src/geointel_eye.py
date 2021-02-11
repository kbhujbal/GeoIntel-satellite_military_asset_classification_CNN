#!/usr/bin/env python3
"""
GEOINTEL EYE - Main Satellite Reconnaissance Inference Engine
===============================================================

The GeoIntel Eye is the core reconnaissance module of the GeoIntel system,
designed for detecting small military assets in high-resolution satellite imagery.

This module solves THE CRITICAL SMALL OBJECT PROBLEM in satellite imagery:
- Satellite images: 4000x4000 pixels (or larger)
- Military vehicles: ~20x20 pixels (tanks, trucks, cargo)
- Standard YOLO resize to 640x640 = DESTROYS small object features

SOLUTION: SAHI (Slicing Aided Hyper Inference)
- Slice massive images into overlapping 512x512 tiles
- Run detection on each tile
- Merge predictions using NMS to handle overlapping regions
- Output: Full battlefield map with all detected assets

Usage:
    # Command line
    python -m src.geointel_eye --image satellite_scan.jpg --output battlefield_map.jpg

    # Python API
    from src.geointel_eye import GeoIntelEye

    eye = GeoIntelEye(model_path="models/geointel_best.pt")
    detections = eye.scan(
        image_path="satellite_scan.jpg",
        output_path="final_map_with_tanks.jpg"
    )

Key Features:
    - SAHI's get_sliced_prediction for tile-based inference
    - Automatic tile overlap handling (20% default)
    - Multi-class military vehicle detection
    - GeoJSON export for GIS integration
    - Comparison mode: Standard vs SAHI detection
"""

import os
import sys
import yaml
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class MilitaryAsset:
    """Detected military asset with full metadata."""
    asset_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x_min, y_min, x_max, y_max
    center: Tuple[float, float]
    area_pixels: int
    detection_method: str  # "sahi" or "standard"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_geojson_feature(self, image_width: int, image_height: int) -> dict:
        """Convert to GeoJSON feature (normalized coordinates)."""
        x_min, y_min, x_max, y_max = self.bbox
        return {
            "type": "Feature",
            "properties": {
                "asset_id": self.asset_id,
                "class": self.class_name,
                "confidence": round(self.confidence, 4),
                "detection_method": self.detection_method
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [x_min / image_width, y_min / image_height],
                    [x_max / image_width, y_min / image_height],
                    [x_max / image_width, y_max / image_height],
                    [x_min / image_width, y_max / image_height],
                    [x_min / image_width, y_min / image_height]
                ]]
            }
        }


@dataclass
class ScanResult:
    """Complete scan result with all detections and metadata."""
    image_path: str
    image_size: Tuple[int, int]
    total_detections: int
    detections_by_class: Dict[str, int]
    assets: List[MilitaryAsset]
    inference_time_seconds: float
    method: str
    tile_info: Optional[Dict] = None

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_size": self.image_size,
            "total_detections": self.total_detections,
            "detections_by_class": self.detections_by_class,
            "assets": [a.to_dict() for a in self.assets],
            "inference_time_seconds": self.inference_time_seconds,
            "method": self.method,
            "tile_info": self.tile_info
        }

    def to_geojson(self) -> dict:
        """Export as GeoJSON FeatureCollection."""
        return {
            "type": "FeatureCollection",
            "properties": {
                "source": self.image_path,
                "total_detections": self.total_detections,
                "scan_method": self.method,
                "timestamp": datetime.now().isoformat()
            },
            "features": [
                asset.to_geojson_feature(*self.image_size)
                for asset in self.assets
            ]
        }


class GeoIntelEye:
    """
    The All-Seeing Eye - SAHI-powered Military Asset Detection System.

    Implements sliding window inference using SAHI to detect small military
    vehicles that standard YOLO would miss due to aggressive downscaling.
    """

    # Class colors for visualization (BGR format)
    CLASS_COLORS = {
        'tank': (0, 0, 255),           # Red
        'truck': (0, 255, 0),          # Green
        'cargo': (255, 0, 0),          # Blue
        'military_vehicle': (0, 255, 255),  # Yellow
        'default': (255, 255, 0)       # Cyan
    }

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        config_path: str = "config/config.yaml",
        device: str = ""
    ):
        """
        Initialize the GeoIntel Eye.

        Args:
            model_path: Path to YOLO model weights
            config_path: Path to configuration file
            device: Inference device ("cuda", "cpu", or "" for auto)
        """
        self.config = self._load_config(config_path)
        self.model_path = model_path
        self.device = device

        # SAHI configuration
        sahi_config = self.config.get('inference', {}).get('sahi', {})
        self.slice_size = sahi_config.get('slice_size', 512)
        self.overlap_ratio = sahi_config.get('overlap_ratio', 0.2)
        self.confidence_threshold = sahi_config.get('sahi_confidence',
            self.config.get('model', {}).get('confidence_threshold', 0.25))
        self.iou_threshold = sahi_config.get('sahi_iou',
            self.config.get('model', {}).get('iou_threshold', 0.45))
        self.postprocess_type = sahi_config.get('postprocess_type', 'NMS')
        self.postprocess_match_threshold = sahi_config.get('postprocess_match_threshold', 0.5)

        # Class names
        self.class_names = self.config.get('dataset', {}).get('classes',
            ['tank', 'truck', 'cargo', 'military_vehicle'])

        # Initialize SAHI detection model
        self.detection_model = None
        self._setup_model()

        logger.info("=" * 60)
        logger.info("GEOINTEL EYE Initialized")
        logger.info("=" * 60)
        logger.info(f"Model: {model_path}")
        logger.info(f"Slice Size: {self.slice_size}x{self.slice_size}")
        logger.info(f"Overlap Ratio: {self.overlap_ratio}")
        logger.info(f"Confidence Threshold: {self.confidence_threshold}")
        logger.info("=" * 60)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        logger.warning(f"Config not found: {config_path}, using defaults")
        return {}

    def _setup_model(self):
        """Initialize the SAHI detection model wrapper."""
        try:
            from sahi import AutoDetectionModel
        except ImportError:
            raise ImportError(
                "SAHI not installed. Install with: pip install sahi"
            )

        logger.info(f"Loading detection model: {self.model_path}")

        # SAHI wraps YOLO models for sliced inference
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            device=self.device if self.device else None
        )

        logger.info("Detection model loaded successfully")

    def scan(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        export_geojson: bool = True,
        visualize: bool = True,
        compare_standard: bool = False
    ) -> ScanResult:
        """
        Scan a satellite image for military assets using SAHI.

        This is the PRIMARY method - uses sliced inference to detect
        small objects that standard inference would miss.

        Args:
            image_path: Path to input satellite image
            output_path: Path for output visualization (default: auto-generated)
            export_geojson: Export detections as GeoJSON
            visualize: Generate visualization image
            compare_standard: Also run standard inference for comparison

        Returns:
            ScanResult with all detected military assets
        """
        try:
            from sahi.predict import get_sliced_prediction
        except ImportError:
            raise ImportError("SAHI not installed. Run: pip install sahi")

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info("=" * 60)
        logger.info(f"SCANNING: {image_path.name}")
        logger.info("=" * 60)

        # Load image to get dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        height, width = image.shape[:2]
        logger.info(f"Image size: {width}x{height} pixels")

        # Calculate tile statistics
        n_tiles_x = int(np.ceil((width - self.slice_size * self.overlap_ratio) /
                                (self.slice_size * (1 - self.overlap_ratio))))
        n_tiles_y = int(np.ceil((height - self.slice_size * self.overlap_ratio) /
                                (self.slice_size * (1 - self.overlap_ratio))))
        total_tiles = n_tiles_x * n_tiles_y

        logger.info(f"Tile grid: {n_tiles_x}x{n_tiles_y} = {total_tiles} tiles")
        logger.info(f"Tile size: {self.slice_size}x{self.slice_size}")
        logger.info(f"Overlap: {int(self.overlap_ratio * 100)}%")

        # ============================================================
        # CORE SAHI INFERENCE - get_sliced_prediction
        # ============================================================
        # This is the key function that solves the small object problem!
        # Instead of resizing 4000x4000 -> 640x640 (destroying small objects),
        # it processes overlapping 512x512 tiles and merges results.
        # ============================================================

        logger.info("Running SAHI sliced inference...")
        start_time = time.time()

        result = get_sliced_prediction(
            image=str(image_path),
            detection_model=self.detection_model,

            # Tile configuration
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,

            # Post-processing to merge overlapping detections
            postprocess_type=self.postprocess_type,  # "NMS" or "GREEDYNMM"
            postprocess_match_metric="IOU",
            postprocess_match_threshold=self.postprocess_match_threshold,
            postprocess_class_agnostic=False,

            # Performance options
            perform_standard_pred=False,  # Don't also run standard inference
            verbose=2  # Show progress
        )

        inference_time = time.time() - start_time
        logger.info(f"Inference complete in {inference_time:.2f}s")

        # ============================================================
        # Process SAHI results into MilitaryAsset objects
        # ============================================================

        assets = []
        detections_by_class = {cls: 0 for cls in self.class_names}

        for idx, pred in enumerate(result.object_prediction_list):
            bbox = pred.bbox
            x_min, y_min, x_max, y_max = (
                int(bbox.minx), int(bbox.miny),
                int(bbox.maxx), int(bbox.maxy)
            )

            class_name = pred.category.name
            if class_name in detections_by_class:
                detections_by_class[class_name] += 1

            asset = MilitaryAsset(
                asset_id=idx,
                class_id=pred.category.id,
                class_name=class_name,
                confidence=pred.score.value,
                bbox=(x_min, y_min, x_max, y_max),
                center=((x_min + x_max) / 2, (y_min + y_max) / 2),
                area_pixels=(x_max - x_min) * (y_max - y_min),
                detection_method="sahi"
            )
            assets.append(asset)

        # Build result object
        scan_result = ScanResult(
            image_path=str(image_path),
            image_size=(width, height),
            total_detections=len(assets),
            detections_by_class=detections_by_class,
            assets=assets,
            inference_time_seconds=round(inference_time, 3),
            method="sahi_sliced",
            tile_info={
                "tile_size": self.slice_size,
                "overlap_ratio": self.overlap_ratio,
                "grid": f"{n_tiles_x}x{n_tiles_y}",
                "total_tiles": total_tiles
            }
        )

        # Log detection summary
        logger.info("-" * 40)
        logger.info("DETECTION SUMMARY")
        logger.info("-" * 40)
        logger.info(f"Total assets detected: {len(assets)}")
        for cls, count in detections_by_class.items():
            if count > 0:
                logger.info(f"  {cls}: {count}")
        logger.info("-" * 40)

        # Generate output path if not specified
        if output_path is None:
            output_dir = Path(self.config.get('output', {}).get('save_dir', 'data/outputs'))
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"scan_{image_path.stem}_{timestamp}.jpg"

        # Save visualization
        if visualize:
            self._visualize_detections(
                image, assets, str(output_path),
                title=f"GeoIntel Scan - {len(assets)} assets detected"
            )
            logger.info(f"Visualization saved: {output_path}")

        # Export GeoJSON
        if export_geojson:
            geojson_path = Path(output_path).with_suffix('.geojson')
            with open(geojson_path, 'w') as f:
                json.dump(scan_result.to_geojson(), f, indent=2)
            logger.info(f"GeoJSON exported: {geojson_path}")

        # Optional: Compare with standard inference
        if compare_standard:
            logger.info("\nRunning comparison with standard inference...")
            standard_result = self.scan_standard(image_path)
            self._print_comparison(scan_result, standard_result)

        return scan_result

    def scan_standard(self, image_path: str) -> ScanResult:
        """
        Run standard (non-sliced) YOLO inference for comparison.

        This demonstrates WHY SAHI is necessary - standard inference
        misses small objects due to aggressive image resizing.

        Args:
            image_path: Path to input image

        Returns:
            ScanResult from standard inference
        """
        try:
            from sahi.predict import get_prediction
        except ImportError:
            raise ImportError("SAHI not installed")

        logger.info("Running STANDARD inference (no tiling)...")

        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        start_time = time.time()

        result = get_prediction(
            image=str(image_path),
            detection_model=self.detection_model,
            shift_amount=[0, 0],
            full_shape=None,
            postprocess=None
        )

        inference_time = time.time() - start_time

        assets = []
        detections_by_class = {cls: 0 for cls in self.class_names}

        for idx, pred in enumerate(result.object_prediction_list):
            bbox = pred.bbox
            class_name = pred.category.name

            if class_name in detections_by_class:
                detections_by_class[class_name] += 1

            asset = MilitaryAsset(
                asset_id=idx,
                class_id=pred.category.id,
                class_name=class_name,
                confidence=pred.score.value,
                bbox=(int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)),
                center=((bbox.minx + bbox.maxx) / 2, (bbox.miny + bbox.maxy) / 2),
                area_pixels=int((bbox.maxx - bbox.minx) * (bbox.maxy - bbox.miny)),
                detection_method="standard"
            )
            assets.append(asset)

        return ScanResult(
            image_path=str(image_path),
            image_size=(width, height),
            total_detections=len(assets),
            detections_by_class=detections_by_class,
            assets=assets,
            inference_time_seconds=round(inference_time, 3),
            method="standard",
            tile_info=None
        )

    def _print_comparison(self, sahi_result: ScanResult, standard_result: ScanResult):
        """Print comparison between SAHI and standard inference."""
        logger.info("=" * 60)
        logger.info("SAHI vs STANDARD INFERENCE COMPARISON")
        logger.info("=" * 60)
        logger.info(f"{'Metric':<25} {'SAHI':<15} {'Standard':<15}")
        logger.info("-" * 60)
        logger.info(f"{'Total Detections':<25} {sahi_result.total_detections:<15} {standard_result.total_detections:<15}")
        logger.info(f"{'Inference Time (s)':<25} {sahi_result.inference_time_seconds:<15.2f} {standard_result.inference_time_seconds:<15.2f}")

        for cls in self.class_names:
            sahi_count = sahi_result.detections_by_class.get(cls, 0)
            std_count = standard_result.detections_by_class.get(cls, 0)
            if sahi_count > 0 or std_count > 0:
                logger.info(f"{cls:<25} {sahi_count:<15} {std_count:<15}")

        improvement = sahi_result.total_detections - standard_result.total_detections
        if improvement > 0:
            pct = (improvement / max(standard_result.total_detections, 1)) * 100
            logger.info("-" * 60)
            logger.info(f"SAHI detected {improvement} MORE assets ({pct:.1f}% improvement)")
        logger.info("=" * 60)

    def _visualize_detections(
        self,
        image: np.ndarray,
        assets: List[MilitaryAsset],
        output_path: str,
        title: Optional[str] = None
    ):
        """
        Draw bounding boxes on the image.

        Args:
            image: Input image (BGR)
            assets: List of detected assets
            output_path: Path to save visualization
            title: Optional title to add to image
        """
        vis_image = image.copy()

        for asset in assets:
            x_min, y_min, x_max, y_max = asset.bbox
            color = self.CLASS_COLORS.get(asset.class_name, self.CLASS_COLORS['default'])

            # Draw bounding box
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)

            # Draw label background
            label = f"{asset.class_name} {asset.confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            cv2.rectangle(
                vis_image,
                (x_min, y_min - label_height - 10),
                (x_min + label_width, y_min),
                color, -1
            )

            # Draw label text
            cv2.putText(
                vis_image, label,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # Add title
        if title:
            cv2.putText(
                vis_image, title,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        # Add legend
        y_offset = 60
        for cls, color in self.CLASS_COLORS.items():
            if cls == 'default':
                continue
            cv2.rectangle(vis_image, (10, y_offset), (30, y_offset + 20), color, -1)
            cv2.putText(
                vis_image, cls.capitalize(),
                (40, y_offset + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            y_offset += 30

        cv2.imwrite(output_path, vis_image)

    def batch_scan(
        self,
        image_dir: str,
        output_dir: Optional[str] = None,
        pattern: str = "*.jpg"
    ) -> List[ScanResult]:
        """
        Scan multiple satellite images.

        Args:
            image_dir: Directory containing images
            output_dir: Output directory for results
            pattern: Glob pattern for images

        Returns:
            List of ScanResult objects
        """
        from glob import glob

        image_dir = Path(image_dir)
        images = list(image_dir.glob(pattern))
        images.extend(image_dir.glob(pattern.replace('.jpg', '.png')))
        images.extend(image_dir.glob(pattern.replace('.jpg', '.tif')))

        logger.info(f"Found {len(images)} images to process")

        if output_dir is None:
            output_dir = image_dir / "scan_results"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, img_path in enumerate(images, 1):
            logger.info(f"\nProcessing [{i}/{len(images)}]: {img_path.name}")
            try:
                output_path = output_dir / f"scan_{img_path.stem}.jpg"
                result = self.scan(
                    str(img_path),
                    output_path=str(output_path)
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")

        # Summary
        total_assets = sum(r.total_detections for r in results)
        logger.info("\n" + "=" * 60)
        logger.info("BATCH SCAN COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Images processed: {len(results)}")
        logger.info(f"Total assets detected: {total_assets}")

        return results


def main():
    """Command line interface for GeoIntel Eye."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GeoIntel Eye - Satellite Military Asset Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan single image
  python -m src.geointel_eye --image satellite.jpg --output detected.jpg

  # Scan with comparison to standard inference
  python -m src.geointel_eye --image satellite.jpg --compare

  # Batch scan directory
  python -m src.geointel_eye --batch-dir images/ --output-dir results/

  # Custom tile size and overlap
  python -m src.geointel_eye --image sat.jpg --tile-size 640 --overlap 0.3
        """
    )

    parser.add_argument("--image", help="Path to input satellite image")
    parser.add_argument("--output", help="Output path for visualization")
    parser.add_argument("--model", default="yolov8m.pt", help="Path to YOLO model")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for slicing")
    parser.add_argument("--overlap", type=float, default=0.2, help="Tile overlap ratio")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--compare", action="store_true", help="Compare SAHI vs standard inference")
    parser.add_argument("--batch-dir", help="Directory for batch processing")
    parser.add_argument("--output-dir", help="Output directory for batch results")
    parser.add_argument("--device", default="", help="Device (cuda/cpu/'')")
    parser.add_argument("--no-visualize", action="store_true", help="Skip visualization")
    parser.add_argument("--no-geojson", action="store_true", help="Skip GeoJSON export")

    args = parser.parse_args()

    # Initialize GeoIntel Eye
    eye = GeoIntelEye(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )

    # Override config with CLI args
    eye.slice_size = args.tile_size
    eye.overlap_ratio = args.overlap
    eye.confidence_threshold = args.confidence

    if args.batch_dir:
        # Batch mode
        eye.batch_scan(
            image_dir=args.batch_dir,
            output_dir=args.output_dir
        )
    elif args.image:
        # Single image mode
        result = eye.scan(
            image_path=args.image,
            output_path=args.output,
            visualize=not args.no_visualize,
            export_geojson=not args.no_geojson,
            compare_standard=args.compare
        )

        # Print JSON summary
        print("\n" + json.dumps(result.to_dict(), indent=2))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
