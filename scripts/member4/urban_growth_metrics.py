"""
Member 4: Urban Growth Metrics & Analysis Lead
===============================================
Reads the classified GeoTIFF outputs produced by Member 3 (one per city per year)
and computes the following statistics:

  - Urban area in km² for each city / year
  - Absolute growth (km²) between consecutive decades
  - Percentage growth between consecutive decades

Also produces three publication-ready charts and a short written analysis.

Outputs
-------
  data/urban_growth_metrics.csv     — full metrics table
  data/charts/urban_area_over_time.png
  data/charts/city_comparison_growth.png
  data/charts/urban_area_by_city_year.png

Usage
-----
  cd <project_root>
  python scripts/member4/urban_growth_metrics.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

# =========================================================
# CONFIGURATION
# =========================================================

# Navigate two levels up from scripts/member4/ to reach the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASSIFIED_DIR = PROJECT_ROOT / "data" / "classified"
OUTPUT_DIR     = PROJECT_ROOT / "data"
CHARTS_DIR     = PROJECT_ROOT / "data" / "charts"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Cities and years to process — must match the filenames produced by Member 3
CITIES = ["riverside", "phoenix", "las_vegas", "austin"]
YEARS  = [1990, 2000, 2010, 2020]

# Landsat spatial resolution: 30 m × 30 m = 900 m² per pixel
PIXEL_AREA_M2  = 900
PIXEL_AREA_KM2 = PIXEL_AREA_M2 / 1_000_000    # 0.0009 km²

# Value written to nodata pixels by Member 3's classifier
NODATA_VALUE = 255

# Human-readable city names for chart labels
CITY_LABELS = {
    "riverside": "Riverside",
    "phoenix":   "Phoenix",
    "las_vegas": "Las Vegas",
    "austin":    "Austin",
}

# Consistent color palette — reused across all charts
CITY_COLORS = {
    "riverside": "#2196F3",
    "phoenix":   "#FF5722",
    "las_vegas": "#9C27B0",
    "austin":    "#4CAF50",
}


# =========================================================
# STEP 1 — Count urban pixels and convert to area
# =========================================================

def count_urban_pixels(tif_path: Path) -> int:
    """
    Open a single-band classification GeoTIFF and return the number of
    pixels whose value equals 1 (urban).
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1)   # Band 1 contains the classification labels
    return int(np.sum(data == 1))


def pixels_to_km2(pixel_count: int) -> float:
    """Convert a pixel count to area in km², rounded to 2 decimal places."""
    return round(pixel_count * PIXEL_AREA_KM2, 2)


# =========================================================
# STEP 2 — Build base metrics table
# =========================================================

print("=== Computing Urban Area ===")
records = []

for city in CITIES:
    for year in YEARS:
        filename = f"{city}_{year}_classification.tif"
        tif_path = CLASSIFIED_DIR / filename

        if not tif_path.exists():
            print(f"  [WARNING] Missing file: {tif_path}")
            records.append({
                "city":           city,
                "year":           year,
                "urban_pixels":   None,
                "urban_area_km2": None,
            })
            continue

        urban_px  = count_urban_pixels(tif_path)
        urban_km2 = pixels_to_km2(urban_px)

        print(f"  {city:<12} {year}:  {urban_px:>10,d} urban pixels  →  {urban_km2:>8.2f} km²")
        records.append({
            "city":           city,
            "year":           year,
            "urban_pixels":   urban_px,
            "urban_area_km2": urban_km2,
        })

metrics_df = pd.DataFrame(records)


# =========================================================
# STEP 3 — Calculate growth between consecutive decades
# =========================================================

print("\n=== Calculating Growth ===")
growth_rows = []

for city in CITIES:
    # Sort by year so growth is always computed forward in time
    city_df = (
        metrics_df[metrics_df["city"] == city]
        .sort_values("year")
        .reset_index(drop=True)
    )

    for i, row in city_df.iterrows():
        area = row["urban_area_km2"]
        year = row["year"]

        if i == 0:
            # Baseline year — no previous decade to compare against
            growth_km2 = None
            growth_pct = None
        else:
            prev_area = city_df.loc[i - 1, "urban_area_km2"]

            if prev_area is not None and prev_area > 0 and area is not None:
                # Absolute growth in km²
                growth_km2 = round(area - prev_area, 2)
                # Percentage growth relative to the previous decade
                growth_pct = round(((area - prev_area) / prev_area) * 100, 1)
            else:
                growth_km2 = None
                growth_pct = None

        growth_rows.append({
            "city":           city,
            "year":           year,
            "urban_area_km2": area,
            "growth_km2":     growth_km2,
            "growth_pct":     growth_pct,
        })

growth_df = pd.DataFrame(growth_rows)

# Add a human-readable display column for percentage growth
growth_df["growth_pct_display"] = growth_df["growth_pct"].apply(
    lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
)

# ── Print full table ─────────────────────────────────────────────────────────
print("\n=== Urban Growth Metrics Table ===")
print(growth_df[["city", "year", "urban_area_km2", "growth_km2", "growth_pct_display"]]
      .to_string(index=False))

# ── Export CSV ───────────────────────────────────────────────────────────────
metrics_csv_path = OUTPUT_DIR / "urban_growth_metrics.csv"
growth_df.to_csv(metrics_csv_path, index=False)
print(f"\nMetrics saved to: {metrics_csv_path}")


# =========================================================
# STEP 4 — Charts
# =========================================================

# ── Chart 1: Urban Area vs Year (line chart, one line per city) ─────────────
fig1, ax1 = plt.subplots(figsize=(10, 6))

for city in CITIES:
    city_data = growth_df[growth_df["city"] == city].sort_values("year")

    ax1.plot(
        city_data["year"],
        city_data["urban_area_km2"],
        marker="o",
        linewidth=2.5,
        label=CITY_LABELS[city],
        color=CITY_COLORS[city],
    )

    # Annotate the 2020 endpoint with its area value
    last = city_data.iloc[-1]
    if pd.notna(last["urban_area_km2"]):
        ax1.annotate(
            f"{last['urban_area_km2']:.0f} km²",
            xy=(last["year"], last["urban_area_km2"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=CITY_COLORS[city],
        )

ax1.set_title("Urban Area Over Time (1990–2020)", fontsize=14, fontweight="bold")
ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("Urban Area (km²)", fontsize=12)
ax1.set_xticks(YEARS)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
plt.tight_layout()

chart1_path = CHARTS_DIR / "urban_area_over_time.png"
fig1.savefig(chart1_path, dpi=150)
plt.close(fig1)
print(f"Chart saved: {chart1_path}")


# ── Chart 2: City Comparison — total growth 1990→2020 (bar chart) ───────────
comparison_data = []

for city in CITIES:
    city_data = growth_df[growth_df["city"] == city].sort_values("year")
    a1990 = city_data.loc[city_data["year"] == 1990, "urban_area_km2"].values
    a2020 = city_data.loc[city_data["year"] == 2020, "urban_area_km2"].values

    if len(a1990) and len(a2020) and pd.notna(a1990[0]) and pd.notna(a2020[0]):
        total_growth = round(a2020[0] - a1990[0], 2)
        pct_growth   = round(((a2020[0] - a1990[0]) / a1990[0]) * 100, 1) if a1990[0] > 0 else 0
        comparison_data.append({
            "city":       CITY_LABELS[city],
            "growth_km2": total_growth,
            "growth_pct": pct_growth,
            "color":      CITY_COLORS[city],
        })

comp_df = pd.DataFrame(comparison_data)

fig2, ax2 = plt.subplots(figsize=(8, 6))

if not comp_df.empty:
    bars = ax2.bar(
        comp_df["city"],
        comp_df["growth_km2"],
        color=comp_df["color"].tolist(),
        edgecolor="white",
        linewidth=0.5,
    )

    # Label each bar with its percentage growth above it
    for bar, (_, row) in zip(bars, comp_df.iterrows()):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(comp_df["growth_km2"]) * 0.02,
            f"+{row['growth_pct']:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

ax2.set_title("Total Urban Growth 1990–2020 by City", fontsize=14, fontweight="bold")
ax2.set_xlabel("City", fontsize=12)
ax2.set_ylabel("Urban Area Growth (km²)", fontsize=12)
ax2.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

chart2_path = CHARTS_DIR / "city_comparison_growth.png"
fig2.savefig(chart2_path, dpi=150)
plt.close(fig2)
print(f"Chart saved: {chart2_path}")


# ── Chart 3: Grouped bar — urban area for each city across all years ──────────
fig3, ax3 = plt.subplots(figsize=(12, 6))

x          = np.arange(len(CITIES))
width      = 0.2
year_shades = ["#90CAF9", "#42A5F5", "#1565C0", "#0D47A1"]

for i, year in enumerate(YEARS):
    year_values = []
    for city in CITIES:
        val = growth_df.loc[
            (growth_df["city"] == city) & (growth_df["year"] == year),
            "urban_area_km2"
        ].values
        year_values.append(float(val[0]) if len(val) and pd.notna(val[0]) else 0.0)

    ax3.bar(x + i * width, year_values, width, label=str(year), color=year_shades[i])

ax3.set_title("Urban Area by City and Year", fontsize=14, fontweight="bold")
ax3.set_xlabel("City", fontsize=12)
ax3.set_ylabel("Urban Area (km²)", fontsize=12)
ax3.set_xticks(x + width * 1.5)
ax3.set_xticklabels([CITY_LABELS[c] for c in CITIES])
ax3.legend(title="Year", fontsize=10)
ax3.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

chart3_path = CHARTS_DIR / "urban_area_by_city_year.png"
fig3.savefig(chart3_path, dpi=150)
plt.close(fig3)
print(f"Chart saved: {chart3_path}")


# =========================================================
# STEP 5 — Short analysis
# =========================================================

print("\n=== Short Analysis ===")

# Which city grew fastest by percentage?
if not comp_df.empty:
    fastest_city   = comp_df.loc[comp_df["growth_pct"].idxmax()]
    largest_city   = comp_df.loc[comp_df["growth_km2"].idxmax()]
    print(f"  Fastest % growth (1990–2020): {fastest_city['city']} (+{fastest_city['growth_pct']:.1f}%)")
    print(f"  Largest absolute growth:      {largest_city['city']} (+{largest_city['growth_km2']:.1f} km²)")

# Which decade had the most total urban expansion across all cities?
decade_growth = {}
for yr_start, yr_end in [(1990, 2000), (2000, 2010), (2010, 2020)]:
    total = 0.0
    for city in CITIES:
        s = growth_df.loc[(growth_df["city"] == city) & (growth_df["year"] == yr_start), "urban_area_km2"].values
        e = growth_df.loc[(growth_df["city"] == city) & (growth_df["year"] == yr_end),   "urban_area_km2"].values
        if len(s) and len(e) and pd.notna(s[0]) and pd.notna(e[0]):
            total += e[0] - s[0]
    decade_growth[f"{yr_start}–{yr_end}"] = round(total, 2)

most_active_decade = max(decade_growth, key=lambda k: decade_growth[k])
print(f"  Most active decade: {most_active_decade} "
      f"(+{decade_growth[most_active_decade]:.1f} km² total across all cities)")

print("\n  Decade breakdown (all cities combined):")
for decade, growth in decade_growth.items():
    print(f"    {decade}: +{growth:.1f} km²")

print("\nDone. All outputs written to data/")
