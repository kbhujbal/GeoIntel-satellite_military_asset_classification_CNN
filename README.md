# GeoIntel - Satellite Military Asset Classification

> **SAHI-Powered Reconnaissance System for Small Object Detection in Satellite Imagery**

## Mission

Identify small military vehicles (Tanks, Trucks, Cargo) in high-resolution satellite imagery using **Slicing Aided Hyper Inference (SAHI)** to solve the critical **Small Object Problem**.

## The Small Object Problem

| Challenge | Standard Approach | GeoIntel Solution |
|-----------|-------------------|-------------------|
| Satellite Image | 4000×4000 px | Process as tiles |
| Military Vehicle | ~20×20 px | Preserved at full resolution |
| YOLO Input | Resize to 640×640 | 512×512 overlapping tiles |
| Result | **Objects destroyed** | **Objects detected** |

```
Standard YOLO:     4000px → 640px = 84% resolution loss
                   20px tank → 3px blob (undetectable!)

GeoIntel + SAHI:   4000px → 64 tiles @ 512px each
                   20px tank stays 20px (detectable!)
```

## Quick Start

```bash
# 1. Clone and setup
cd GeoIntel
pip install -r requirements.txt

# 2. Verify your dataset
python -m src.data_loader --data-dir data/raw

# 3. Run SAHI inference on a satellite image
python -m src.geointel_eye --image satellite.jpg --output final_map_with_tanks.jpg
```

## Project Structure

```
GeoIntel/
├── config/
│   └── config.yaml          # Tile sizes, overlap ratios, model params
├── src/
│   ├── data_loader.py       # MVRSD dataset verification & loading
│   ├── tiling_utils.py      # Manual image slicing utilities
│   ├── trainer.py           # YOLOv8 fine-tuning pipeline
│   └── geointel_eye.py      # 🎯 Main SAHI inference engine
├── notebooks/
│   └── Exploration.ipynb    # Dataset visualization & analysis
├── data/
│   ├── raw/                 # Place MVRSD dataset here
│   │   ├── images/
│   │   └── labels/
│   ├── processed/           # Train/val/test splits
│   └── outputs/             # Detection results
├── models/                  # Trained model weights
└── requirements.txt
```

## Core Component: GeoIntel Eye

The heart of the system - `src/geointel_eye.py` - uses SAHI's `get_sliced_prediction` to detect small objects:

```python
from src.geointel_eye import GeoIntelEye

# Initialize with YOLOv8-Medium
eye = GeoIntelEye(model_path="models/geointel_best.pt")

# Scan satellite imagery
result = eye.scan(
    image_path="satellite_scan.jpg",
    output_path="final_map_with_tanks.jpg",
    compare_standard=True  # See SAHI vs standard comparison
)

print(f"Detected {result.total_detections} military assets")
print(f"Tanks: {result.detections_by_class['tank']}")
```

### Key SAHI Parameters

```yaml
# config/config.yaml
tiling:
  slice_height: 512        # Tile size (px)
  slice_width: 512
  overlap_height_ratio: 0.2  # 20% overlap
  overlap_width_ratio: 0.2

inference:
  sahi:
    postprocess_type: "NMS"  # Merge overlapping detections
    postprocess_match_threshold: 0.5
```

## Dataset: MVRSD

**Military Vehicle Remote Sensing Dataset** - Expected structure:

```
data/raw/
├── images/
│   ├── satellite_001.jpg
│   ├── satellite_002.jpg
│   └── ...
└── labels/              # YOLO format
    ├── satellite_001.txt
    ├── satellite_002.txt
    └── ...
```

**YOLO Annotation Format:**
```
# class_id x_center y_center width height (normalized 0-1)
0 0.5 0.5 0.02 0.02   # Tank at center, 2% of image
1 0.3 0.7 0.03 0.02   # Truck
2 0.8 0.2 0.04 0.03   # Cargo
```

**Classes:**
- 0: Tank
- 1: Truck
- 2: Cargo
- 3: Military Vehicle (generic)

## Training

```bash
# Prepare dataset splits
python -m src.data_loader --data-dir data/raw --prepare-splits --output-dir data/processed

# Train YOLOv8-Medium
python -m src.trainer --data data/processed/data.yaml --epochs 100 --batch-size 16

# Validate
python -m src.trainer --data data/processed/data.yaml --validate-only --model runs/train/best.pt
```

## Inference Modes

### Single Image
```bash
python -m src.geointel_eye --image satellite.jpg --output detected.jpg
```

### Batch Processing
```bash
python -m src.geointel_eye --batch-dir images/ --output-dir results/
```

### Compare SAHI vs Standard
```bash
python -m src.geointel_eye --image satellite.jpg --compare
```

Output:
```
============================================================
SAHI vs STANDARD INFERENCE COMPARISON
============================================================
Metric                    SAHI            Standard
------------------------------------------------------------
Total Detections          47              12
Inference Time (s)        8.32            0.45
tank                      23              5
truck                     18              6
cargo                     6               1
------------------------------------------------------------
SAHI detected 35 MORE assets (291.7% improvement)
============================================================
```

## Outputs

- **Visualization**: `final_map_with_tanks.jpg` - Annotated satellite image
- **GeoJSON**: `scan_results.geojson` - For GIS integration
- **JSON**: Detection metadata with confidence scores

## Configuration

Edit `config/config.yaml` for your use case:

```yaml
model:
  architecture: "yolov8m"      # YOLOv8 Medium
  confidence_threshold: 0.25
  iou_threshold: 0.45

tiling:
  slice_height: 512           # Adjust for your imagery
  slice_width: 512
  overlap_height_ratio: 0.2   # Increase for dense scenes
```

## API Reference

### GeoIntelEye

```python
class GeoIntelEye:
    def scan(
        image_path: str,
        output_path: str = None,
        export_geojson: bool = True,
        visualize: bool = True,
        compare_standard: bool = False
    ) -> ScanResult

    def scan_standard(image_path: str) -> ScanResult

    def batch_scan(image_dir: str, output_dir: str = None) -> List[ScanResult]
```

### ScanResult

```python
@dataclass
class ScanResult:
    image_path: str
    image_size: Tuple[int, int]
    total_detections: int
    detections_by_class: Dict[str, int]
    assets: List[MilitaryAsset]
    inference_time_seconds: float
    method: str  # "sahi_sliced" or "standard"
    tile_info: Dict  # tile_size, overlap, grid dimensions
```

## Why SAHI?

SAHI (Slicing Aided Hyper Inference) by Fatih Akyon:
- Paper: [arxiv.org/abs/2202.06934](https://arxiv.org/abs/2202.06934)
- GitHub: [github.com/obss/sahi](https://github.com/obss/sahi)

**Key Insight**: Instead of downscaling large images (destroying small objects), process overlapping tiles at full resolution and merge predictions.

## License

MIT License

---

**GeoIntel** - *"See what others miss"*
