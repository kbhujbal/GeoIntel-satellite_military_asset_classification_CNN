"""
GeoIntel Model Trainer
======================

Fine-tune YOLOv8 model on the MVRSD dataset for military vehicle detection.
Optimized for small object detection in satellite imagery.

Usage:
    from src.trainer import GeoIntelTrainer

    trainer = GeoIntelTrainer()
    trainer.train(data_yaml="data/processed/data.yaml", epochs=100)
    trainer.validate()
    trainer.export(format="onnx")
"""

import os
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    model: str = "yolov8m.pt"
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    patience: int = 20
    optimizer: str = "AdamW"
    lr0: float = 0.001
    weight_decay: float = 0.0005
    device: str = ""  # Auto-select
    workers: int = 8
    project: str = "runs/train"
    name: str = "geointel"
    exist_ok: bool = True
    pretrained: bool = True
    verbose: bool = True
    seed: int = 42


class GeoIntelTrainer:
    """
    Handles training YOLOv8 models for military vehicle detection.

    Optimizations for small objects:
    - Uses YOLOv8m (medium) for better small object features
    - Configures appropriate augmentation for satellite imagery
    - Implements multi-scale training
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the trainer.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.training_config = self._parse_training_config()
        self.model = None
        self.results = None

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        logger.warning(f"Config not found at {config_path}, using defaults")
        return {}

    def _parse_training_config(self) -> TrainingConfig:
        """Parse training configuration from loaded config."""
        train_cfg = self.config.get('training', {})
        model_cfg = self.config.get('model', {})

        return TrainingConfig(
            model=model_cfg.get('pretrained_weights', 'yolov8m.pt'),
            epochs=train_cfg.get('epochs', 100),
            batch_size=train_cfg.get('batch_size', 16),
            img_size=train_cfg.get('img_size', 640),
            patience=train_cfg.get('patience', 20),
            optimizer=train_cfg.get('optimizer', 'AdamW'),
            lr0=train_cfg.get('learning_rate', 0.001),
            weight_decay=train_cfg.get('weight_decay', 0.0005)
        )

    def setup_model(self, model_path: Optional[str] = None) -> Any:
        """
        Initialize the YOLO model.

        Args:
            model_path: Optional path to custom weights

        Returns:
            YOLO model instance
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

        weights = model_path or self.training_config.model
        logger.info(f"Loading model: {weights}")

        self.model = YOLO(weights)
        return self.model

    def train(
        self,
        data_yaml: str,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        img_size: Optional[int] = None,
        resume: bool = False,
        **kwargs
    ) -> Any:
        """
        Train the model on the MVRSD dataset.

        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image size for training
            resume: Resume from last checkpoint
            **kwargs: Additional training arguments

        Returns:
            Training results
        """
        if self.model is None:
            self.setup_model()

        # Validate data.yaml exists
        if not Path(data_yaml).exists():
            raise FileNotFoundError(f"Data config not found: {data_yaml}")

        # Build training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs or self.training_config.epochs,
            'batch': batch_size or self.training_config.batch_size,
            'imgsz': img_size or self.training_config.img_size,
            'patience': self.training_config.patience,
            'optimizer': self.training_config.optimizer,
            'lr0': self.training_config.lr0,
            'weight_decay': self.training_config.weight_decay,
            'device': self.training_config.device or None,
            'workers': self.training_config.workers,
            'project': self.training_config.project,
            'name': f"{self.training_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'exist_ok': self.training_config.exist_ok,
            'pretrained': self.training_config.pretrained,
            'verbose': self.training_config.verbose,
            'seed': self.training_config.seed,
            'resume': resume,

            # Small object optimizations
            'mosaic': 1.0,  # Mosaic augmentation helps with small objects
            'mixup': 0.1,   # Light mixup
            'copy_paste': 0.1,  # Copy-paste augmentation

            # Satellite imagery specific
            'degrees': 180.0,  # Full rotation (satellite images have no fixed orientation)
            'translate': 0.2,
            'scale': 0.9,  # More scale variation
            'fliplr': 0.5,
            'flipud': 0.5,  # Vertical flip (useful for satellite)

            # Loss weights - emphasize small objects
            'box': 7.5,  # Box loss weight
            'cls': 0.5,  # Classification loss weight
            'dfl': 1.5,  # Distribution focal loss weight
        }

        # Override with any additional kwargs
        train_args.update(kwargs)

        logger.info("=" * 60)
        logger.info("GeoIntel Training Configuration")
        logger.info("=" * 60)
        for key, value in train_args.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

        # Start training
        logger.info("Starting training...")
        self.results = self.model.train(**train_args)

        logger.info("Training complete!")
        return self.results

    def validate(
        self,
        data_yaml: Optional[str] = None,
        model_path: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Validate the trained model.

        Args:
            data_yaml: Path to data.yaml (uses training data if not specified)
            model_path: Path to model weights
            **kwargs: Additional validation arguments

        Returns:
            Validation metrics
        """
        if model_path:
            self.setup_model(model_path)
        elif self.model is None:
            raise ValueError("No model loaded. Call setup_model() or provide model_path")

        val_args = {
            'data': data_yaml,
            'imgsz': self.training_config.img_size,
            'batch': self.training_config.batch_size,
            'verbose': True,
            'plots': True,
        }
        val_args.update(kwargs)

        logger.info("Running validation...")
        metrics = self.model.val(**val_args)

        # Log key metrics
        logger.info("=" * 60)
        logger.info("Validation Results")
        logger.info("=" * 60)
        logger.info(f"  mAP50: {metrics.box.map50:.4f}")
        logger.info(f"  mAP50-95: {metrics.box.map:.4f}")
        logger.info(f"  Precision: {metrics.box.mp:.4f}")
        logger.info(f"  Recall: {metrics.box.mr:.4f}")
        logger.info("=" * 60)

        return {
            'map50': metrics.box.map50,
            'map50_95': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr,
            'per_class_ap50': metrics.box.ap50.tolist() if hasattr(metrics.box, 'ap50') else None
        }

    def export(
        self,
        model_path: Optional[str] = None,
        format: str = "onnx",
        **kwargs
    ) -> str:
        """
        Export the model to various formats.

        Args:
            model_path: Path to model weights
            format: Export format (onnx, torchscript, tflite, etc.)
            **kwargs: Additional export arguments

        Returns:
            Path to exported model
        """
        if model_path:
            self.setup_model(model_path)
        elif self.model is None:
            raise ValueError("No model loaded")

        export_args = {
            'format': format,
            'imgsz': self.training_config.img_size,
            'simplify': True,  # Simplify ONNX model
            'dynamic': False,  # Static shape for inference optimization
        }
        export_args.update(kwargs)

        logger.info(f"Exporting model to {format} format...")
        export_path = self.model.export(**export_args)

        logger.info(f"Model exported to: {export_path}")
        return export_path

    def benchmark(
        self,
        model_path: Optional[str] = None,
        data_yaml: Optional[str] = None
    ) -> Dict:
        """
        Benchmark model performance.

        Args:
            model_path: Path to model weights
            data_yaml: Path to data.yaml for benchmark

        Returns:
            Benchmark results including speed metrics
        """
        if model_path:
            self.setup_model(model_path)
        elif self.model is None:
            raise ValueError("No model loaded")

        try:
            from ultralytics.utils.benchmarks import benchmark
        except ImportError:
            logger.warning("Benchmark utility not available")
            return {}

        logger.info("Running benchmark...")
        results = benchmark(
            model=self.model,
            data=data_yaml,
            imgsz=self.training_config.img_size
        )

        return results


class SmallObjectTrainer(GeoIntelTrainer):
    """
    Specialized trainer with additional optimizations for small object detection.
    Implements techniques specifically designed for detecting tiny military vehicles.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path)

    def train_with_tiling(
        self,
        data_yaml: str,
        tile_size: int = 512,
        **kwargs
    ) -> Any:
        """
        Train with tiled dataset for better small object detection.

        This method assumes the dataset has been pre-tiled using tiling_utils.
        Each tile is treated as an independent training sample.

        Args:
            data_yaml: Path to tiled data.yaml
            tile_size: Size of tiles (for reference)
            **kwargs: Additional training arguments
        """
        # For tiled training, we can use larger image sizes
        # since tiles are already manageable
        kwargs.setdefault('imgsz', min(tile_size, 640))

        # Adjust batch size for potentially more tiles
        kwargs.setdefault('batch', 32)

        logger.info(f"Training with tiled dataset (tile_size={tile_size})")
        return self.train(data_yaml, **kwargs)

    def train_multi_scale(
        self,
        data_yaml: str,
        scales: List[int] = [480, 640, 800],
        **kwargs
    ) -> Any:
        """
        Train with multi-scale augmentation for better scale invariance.

        Args:
            data_yaml: Path to data.yaml
            scales: List of image sizes to train with
            **kwargs: Additional training arguments
        """
        # YOLOv8 supports multi-scale training natively
        # We set scale augmentation high
        kwargs.setdefault('scale', 0.9)  # +/- 90% scale variation

        # Use middle scale as base
        base_size = scales[len(scales) // 2]
        kwargs.setdefault('imgsz', base_size)

        logger.info(f"Training with multi-scale augmentation (base_size={base_size})")
        return self.train(data_yaml, **kwargs)


def create_training_script(output_path: str = "train_geointel.py"):
    """
    Generate a standalone training script.

    Args:
        output_path: Where to save the script
    """
    script_content = '''#!/usr/bin/env python3
"""
GeoIntel Training Script
========================
Run: python train_geointel.py --data data/processed/data.yaml --epochs 100
"""

import argparse
from src.trainer import GeoIntelTrainer

def main():
    parser = argparse.ArgumentParser(description="Train GeoIntel model")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--model", default="yolov8m.pt", help="Base model")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--device", default="", help="cuda device or cpu")

    args = parser.parse_args()

    trainer = GeoIntelTrainer()
    trainer.setup_model(args.model)
    trainer.train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        resume=args.resume,
        device=args.device
    )

if __name__ == "__main__":
    main()
'''

    with open(output_path, 'w') as f:
        f.write(script_content)
    logger.info(f"Training script created: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GeoIntel Model Training")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--export", choices=['onnx', 'torchscript', 'tflite'])

    args = parser.parse_args()

    trainer = GeoIntelTrainer()
    trainer.setup_model(args.model)

    if args.validate_only:
        trainer.validate(data_yaml=args.data)
    elif args.export:
        trainer.export(format=args.export)
    else:
        trainer.train(
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
