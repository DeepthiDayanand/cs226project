import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

# Navigate two levels up from scripts/member3/ to reach the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# File paths for the composite and classification images.
# Composites are exported to the project data/ root by Member 1.
# Classifications are written to data/classified/ by randomForest_toComposites.py.
composite_path     = PROJECT_ROOT / "data" / "austin_2020_composite.tif"
classification_path = PROJECT_ROOT / "data" / "classified" / "austin_2020_classification.tif"

# read the three bands as float arrays 
with rasterio.open(composite_path) as src:
    red = src.read(1).astype(np.float32)
    green = src.read(2).astype(np.float32)
    blue = src.read(3).astype(np.float32)

rgb = np.dstack([red, green, blue])

# compute display stretch using 2nd and 98th percentiles 
rgb_min = np.nanpercentile(rgb, 2)
rgb_max = np.nanpercentile(rgb, 98)
rgb_display = np.clip((rgb - rgb_min) / (rgb_max - rgb_min), 0, 1)

# open predicted classifcation raster
with rasterio.open(classification_path) as src:
    pred = src.read(1)

# create an empty RGBA overlay image 
overlay = np.zeros((pred.shape[0], pred.shape[1], 4), dtype=np.float32)

# nake urban clearly visible in cyan
overlay[pred == 1] = [0, 1, 1, 0.10]

# creats the plot to visually see with title 
plt.figure(figsize=(10, 10))
plt.imshow(rgb_display)
plt.imshow(overlay)
plt.title("Austin 2020 RGB with Urban Prediction Overlay")
plt.axis("off")
plt.show()