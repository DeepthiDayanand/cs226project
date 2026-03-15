import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml import PipelineModel


# =========================================================
# CONFIGURATION
# =========================================================

# Project root = two levels above this file if stored in scripts/member3/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = DATA_DIR / "urban_rf_model"
OUTPUT_DIR = DATA_DIR / "classified"

# Composite TIFFs to classify
COMPOSITE_FILES = [
    "riverside_1990_composite.tif",
    "riverside_2000_composite.tif",
    "riverside_2010_composite.tif",
    "riverside_2020_composite.tif",
    "phoenix_1990_composite.tif",
    "phoenix_2000_composite.tif",
    "phoenix_2010_composite.tif",
    "phoenix_2020_composite.tif",
    "austin_1990_composite.tif",
    "austin_2000_composite.tif",
    "austin_2010_composite.tif",
    "austin_2020_composite.tif",
    "las_vegas_1990_composite.tif",
    "las_vegas_2000_composite.tif",
    "las_vegas_2010_composite.tif",
    "las_vegas_2020_composite.tif",
]

# Value used for nodata pixels in the output classification raster
OUTPUT_NODATA = 255


# =========================================================
# SPARK SETUP
# =========================================================

spark = SparkSession.builder \
    .appName("ApplyUrbanRFToComposites") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print(f"Loading trained model from: {MODEL_PATH}")
model = PipelineModel.load(str(MODEL_PATH))


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Compute NDVI = (NIR - Red) / (NIR + Red).
    Uses a safe divide to avoid division-by-zero issues.
    """
    denom = nir + red
    ndvi = np.zeros_like(denom, dtype=np.float32)

    valid = denom != 0
    ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]

    # Keep invalid denominator pixels at 0 for now;
    # they should already be excluded by the valid mask if inputs are masked.
    return ndvi


def classify_composite(tif_path: Path, output_path: Path) -> None:
    """
    Read a 4-band composite GeoTIFF, compute NDVI, apply the saved Spark
    Random Forest model, then write a 1-band classification GeoTIFF.
    """
    print(f"\n=== Classifying: {tif_path.name} ===")

    with rasterio.open(tif_path) as src:
        # Read as masked arrays so we can exclude invalid pixels
        # Assumes band order: red, green, blue, nir
        red = src.read(1, masked=True).astype("float32")
        green = src.read(2, masked=True).astype("float32")
        blue = src.read(3, masked=True).astype("float32")
        nir = src.read(4, masked=True).astype("float32")

        height, width = red.shape
        print(f"Raster shape: {height} x {width}")

        # Build a valid-pixel mask:
        # pixel is valid only if all 4 bands are unmasked and finite
        valid_mask = (
            ~red.mask & ~green.mask & ~blue.mask & ~nir.mask &
            np.isfinite(red.filled(np.nan)) &
            np.isfinite(green.filled(np.nan)) &
            np.isfinite(blue.filled(np.nan)) &
            np.isfinite(nir.filled(np.nan))
        )

        valid_count = int(valid_mask.sum())
        total_count = height * width
        print(f"Valid pixels: {valid_count} / {total_count}")

        if valid_count == 0:
            raise ValueError(f"No valid pixels found in {tif_path.name}")

        # Convert masked arrays to normal arrays
        red_data = red.filled(np.nan)
        green_data = green.filled(np.nan)
        blue_data = blue.filled(np.nan)
        nir_data = nir.filled(np.nan)

        # Compute NDVI
        ndvi_data = compute_ndvi(nir_data, red_data)

        # Extract only valid pixels and flatten them into rows
        red_vals = red_data[valid_mask].astype(np.float64)
        green_vals = green_data[valid_mask].astype(np.float64)
        blue_vals = blue_data[valid_mask].astype(np.float64)
        nir_vals = nir_data[valid_mask].astype(np.float64)
        ndvi_vals = ndvi_data[valid_mask].astype(np.float64)

        rows = list(zip(red_vals, green_vals, blue_vals, nir_vals, ndvi_vals))

        schema = StructType([
            StructField("red", DoubleType(), False),
            StructField("green", DoubleType(), False),
            StructField("blue", DoubleType(), False),
            StructField("nir", DoubleType(), False),
            StructField("ndvi", DoubleType(), False),
        ])

        pixel_df = spark.createDataFrame(rows, schema=schema)

        # Apply the saved pipeline model
        predictions = model.transform(pixel_df).select("prediction")

        prediction_values = np.array(
            predictions.rdd.map(lambda r: int(r["prediction"])).collect(),
            dtype=np.uint8
        )

        # Prepare output raster, fill invalid pixels with nodata
        classified = np.full((height, width), OUTPUT_NODATA, dtype=np.uint8)
        classified[valid_mask] = prediction_values

        # Write output GeoTIFF
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=OUTPUT_NODATA,
            compress="lzw"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(classified, 1)

        urban_pixels = int((classified == 1).sum())
        nonurban_pixels = int((classified == 0).sum())
        nodata_pixels = int((classified == OUTPUT_NODATA).sum())

        print(f"Saved: {output_path}")
        print(f"Urban pixels     : {urban_pixels}")
        print(f"Non-urban pixels : {nonurban_pixels}")
        print(f"Nodata pixels    : {nodata_pixels}")


# =========================================================
# MAIN LOOP
# =========================================================

for filename in COMPOSITE_FILES:
    input_path = DATA_DIR / filename

    if not input_path.exists():
        print(f"Skipping missing file: {input_path}")
        continue

    output_name = filename.replace("_composite.tif", "_classification.tif")
    output_path = OUTPUT_DIR / output_name

    classify_composite(input_path, output_path)

print("\nAll requested composites processed.")
spark.stop()
