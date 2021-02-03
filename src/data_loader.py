"""
GeoIntel Data Loader
====================

Handles verification and loading of the MVRSD (Military Vehicle Remote Sensing Dataset).
Since satellite datasets are massive, this module verifies the structure of manually
placed data rather than downloading automatically.

Usage:
    from src.data_loader import MVRSDDataLoader

    loader = MVRSDDataLoader("data/raw")
    if loader.verify_structure():
        train_data, val_data = loader.prepare_splits()
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics about the loaded dataset."""
    total_images: int
    total_annotations: int
    images_with_annotations: int
    class_distribution: Dict[str, int]
    avg_objects_per_image: float
    image_sizes: Dict[str, int]


class MVRSDDataLoader:
    """
    Military Vehicle Remote Sensing Dataset Loader.

    Verifies dataset structure and prepares data for training/inference.
    Expects YOLO format: images/ and labels/ directories with matching filenames.
    """

    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    ANNOTATION_FORMAT = '.txt'  # YOLO format

    def __init__(self, data_dir: str, config_path: str = "config/config.yaml"):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the raw data directory
            config_path: Path to the configuration file
        """
        self.data_dir = Path(data_dir)
        self.config = self._load_config(config_path)
        self.classes = self.config.get('dataset', {}).get('classes', [])

        # Expected subdirectories
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}

    def verify_structure(self) -> bool:
        """
        Verify that the dataset has the expected structure.

        Expected structure:
            data/raw/
            ├── images/
            │   ├── image_001.jpg
            │   ├── image_002.jpg
            │   └── ...
            └── labels/
                ├── image_001.txt
                ├── image_002.txt
                └── ...

        Returns:
            bool: True if structure is valid, False otherwise
        """
        logger.info("=" * 60)
        logger.info("GeoIntel Dataset Verification")
        logger.info("=" * 60)

        # Check if data directory exists
        if not self.data_dir.exists():
            logger.error(f"❌ Data directory not found: {self.data_dir}")
            logger.info(f"   Please create: {self.data_dir}")
            return False

        logger.info(f"✓ Data directory exists: {self.data_dir}")

        # Check for images directory
        if not self.images_dir.exists():
            logger.error(f"❌ Images directory not found: {self.images_dir}")
            self._print_expected_structure()
            return False

        logger.info(f"✓ Images directory exists: {self.images_dir}")

        # Check for labels directory
        if not self.labels_dir.exists():
            logger.error(f"❌ Labels directory not found: {self.labels_dir}")
            self._print_expected_structure()
            return False

        logger.info(f"✓ Labels directory exists: {self.labels_dir}")

        # Count images and labels
        images = self._get_image_files()
        labels = self._get_label_files()

        if len(images) == 0:
            logger.error(f"❌ No images found in {self.images_dir}")
            logger.info(f"   Supported formats: {self.SUPPORTED_IMAGE_FORMATS}")
            return False

        logger.info(f"✓ Found {len(images)} images")
        logger.info(f"✓ Found {len(labels)} label files")

        # Check for matching pairs
        image_stems = {img.stem for img in images}
        label_stems = {lbl.stem for lbl in labels}

        matched = image_stems & label_stems
        images_without_labels = image_stems - label_stems
        labels_without_images = label_stems - image_stems

        logger.info(f"✓ Matched image-label pairs: {len(matched)}")

        if images_without_labels:
            logger.warning(f"⚠ Images without labels: {len(images_without_labels)}")
            if len(images_without_labels) <= 5:
                for name in images_without_labels:
                    logger.warning(f"   - {name}")

        if labels_without_images:
            logger.warning(f"⚠ Labels without images: {len(labels_without_images)}")

        # Validate annotation format
        valid_annotations = self._validate_annotations(list(labels)[:10])
        if not valid_annotations:
            logger.error("❌ Invalid annotation format detected")
            self._print_annotation_format()
            return False

        logger.info("✓ Annotation format is valid (YOLO format)")

        logger.info("=" * 60)
        logger.info("✅ Dataset verification PASSED")
        logger.info("=" * 60)

        return True

    def _get_image_files(self) -> List[Path]:
        """Get all image files from the images directory."""
        images = []
        for ext in self.SUPPORTED_IMAGE_FORMATS:
            images.extend(self.images_dir.glob(f"*{ext}"))
            images.extend(self.images_dir.glob(f"*{ext.upper()}"))
        return sorted(images)

    def _get_label_files(self) -> List[Path]:
        """Get all label files from the labels directory."""
        return sorted(self.labels_dir.glob(f"*{self.ANNOTATION_FORMAT}"))

    def _validate_annotations(self, label_files: List[Path]) -> bool:
        """
        Validate that annotations are in YOLO format.

        YOLO format: class_id x_center y_center width height
        All values normalized to [0, 1]
        """
        for label_file in label_files:
            if not label_file.exists():
                continue

            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    logger.error(f"Invalid format in {label_file}:{line_num}")
                    logger.error(f"   Expected 5 values, got {len(parts)}")
                    return False

                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])

                    # Check if values are normalized
                    for val, name in [(x_center, 'x_center'), (y_center, 'y_center'),
                                     (width, 'width'), (height, 'height')]:
                        if not 0 <= val <= 1:
                            logger.warning(f"Value {name}={val} not normalized in {label_file}:{line_num}")

                except ValueError as e:
                    logger.error(f"Parse error in {label_file}:{line_num}: {e}")
                    return False

        return True

    def get_statistics(self) -> DatasetStats:
        """
        Compute and return dataset statistics.

        Returns:
            DatasetStats object with comprehensive statistics
        """
        images = self._get_image_files()
        labels = self._get_label_files()

        class_counts = {cls: 0 for cls in self.classes}
        total_objects = 0
        images_with_annotations = 0
        image_sizes = {}

        for label_file in labels:
            with open(label_file, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            if lines:
                images_with_annotations += 1
                total_objects += len(lines)

                for line in lines:
                    parts = line.split()
                    if parts:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(self.classes):
                            class_counts[self.classes[class_id]] += 1

        # Sample image sizes (check first 10 images)
        try:
            from PIL import Image
            for img_path in images[:10]:
                with Image.open(img_path) as img:
                    size_key = f"{img.width}x{img.height}"
                    image_sizes[size_key] = image_sizes.get(size_key, 0) + 1
        except ImportError:
            logger.warning("PIL not installed, skipping image size analysis")

        avg_objects = total_objects / max(images_with_annotations, 1)

        return DatasetStats(
            total_images=len(images),
            total_annotations=len(labels),
            images_with_annotations=images_with_annotations,
            class_distribution=class_counts,
            avg_objects_per_image=round(avg_objects, 2),
            image_sizes=image_sizes
        )

    def prepare_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        output_dir: Optional[str] = None,
        seed: int = 42
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Prepare train/val/test splits for YOLO training.

        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            output_dir: Where to create the split directories
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_images, val_images, test_images) paths
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
            "Ratios must sum to 1.0"

        random.seed(seed)

        images = self._get_image_files()
        image_stems = {img.stem: img for img in images}
        labels = self._get_label_files()
        label_stems = {lbl.stem for lbl in labels}

        # Only use images that have labels
        paired_stems = sorted(set(image_stems.keys()) & label_stems)
        random.shuffle(paired_stems)

        n_total = len(paired_stems)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_stems = paired_stems[:n_train]
        val_stems = paired_stems[n_train:n_train + n_val]
        test_stems = paired_stems[n_train + n_val:]

        train_images = [image_stems[s] for s in train_stems]
        val_images = [image_stems[s] for s in val_stems]
        test_images = [image_stems[s] for s in test_stems]

        logger.info(f"Split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

        # Optionally create directory structure for YOLO training
        if output_dir:
            self._create_yolo_structure(
                output_dir, train_stems, val_stems, test_stems, image_stems
            )

        return train_images, val_images, test_images

    def _create_yolo_structure(
        self,
        output_dir: str,
        train_stems: List[str],
        val_stems: List[str],
        test_stems: List[str],
        image_stems: Dict[str, Path]
    ):
        """Create YOLO-compatible directory structure with symlinks."""
        output_path = Path(output_dir)

        for split_name, stems in [('train', train_stems), ('val', val_stems), ('test', test_stems)]:
            split_images = output_path / split_name / 'images'
            split_labels = output_path / split_name / 'labels'
            split_images.mkdir(parents=True, exist_ok=True)
            split_labels.mkdir(parents=True, exist_ok=True)

            for stem in stems:
                # Image
                src_img = image_stems[stem]
                dst_img = split_images / src_img.name
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)

                # Label
                src_lbl = self.labels_dir / f"{stem}.txt"
                dst_lbl = split_labels / f"{stem}.txt"
                if src_lbl.exists() and not dst_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)

        # Create data.yaml for YOLO
        data_yaml = {
            'path': str(output_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {i: name for i, name in enumerate(self.classes)}
        }

        with open(output_path / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        logger.info(f"Created YOLO structure at {output_path}")

    def _print_expected_structure(self):
        """Print the expected directory structure."""
        logger.info("\nExpected directory structure:")
        logger.info(f"""
        {self.data_dir}/
        ├── images/
        │   ├── satellite_001.jpg
        │   ├── satellite_002.png
        │   └── ...
        └── labels/
            ├── satellite_001.txt  (YOLO format)
            ├── satellite_002.txt
            └── ...
        """)

    def _print_annotation_format(self):
        """Print the expected annotation format."""
        logger.info("\nExpected YOLO annotation format:")
        logger.info("""
        Each .txt file should contain one object per line:
        <class_id> <x_center> <y_center> <width> <height>

        Example (labels/satellite_001.txt):
        0 0.5 0.5 0.02 0.02   # Tank at center, 2% of image size
        1 0.3 0.7 0.03 0.02   # Truck at (30%, 70%)
        2 0.8 0.2 0.04 0.03   # Cargo vehicle

        Class IDs:
        0 = tank
        1 = truck
        2 = cargo
        3 = military_vehicle

        All coordinates are normalized [0, 1]
        """)


def main():
    """Run data verification from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify MVRSD dataset structure")
    parser.add_argument("--data-dir", default="data/raw", help="Path to raw data directory")
    parser.add_argument("--prepare-splits", action="store_true", help="Create train/val/test splits")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for splits")

    args = parser.parse_args()

    loader = MVRSDDataLoader(args.data_dir)

    if loader.verify_structure():
        stats = loader.get_statistics()
        logger.info("\nDataset Statistics:")
        logger.info(f"  Total images: {stats.total_images}")
        logger.info(f"  Total annotations: {stats.total_annotations}")
        logger.info(f"  Images with annotations: {stats.images_with_annotations}")
        logger.info(f"  Average objects per image: {stats.avg_objects_per_image}")
        logger.info(f"  Class distribution: {stats.class_distribution}")

        if args.prepare_splits:
            loader.prepare_splits(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
