"""
Member 5: Visualization & Integration Lead
==========================================
Streamlit interactive dashboard for the Urban Expansion Mapping project.
Displays classification maps, urban growth charts, side-by-side comparisons,
and animated time-lapses for four cities across four decades.

Pages
-----
    Home                    — project overview and quick-look summary
    Main Map                — interactive map viewer per city / year
    Side-by-Side            — compare two years or two cities simultaneously
    Urban Growth Trends     — line and bar charts for growth patterns
    Riverside Landcover     — vegetation and water sample trends for Riverside
    Time-Lapse              — animated GIF + manual frame slider
    City Comparison         — bar charts and analysis highlights

Usage
-----
  cd <project_root>
  streamlit run scripts/member5/dashboard.py
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # Must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from PIL import Image, ImageDraw

# =========================================================
# PATHS & CONSTANTS
# =========================================================

# Navigate two levels up from scripts/member5/ to reach the project root
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
CLASSIFIED_DIR = PROJECT_ROOT / "data" / "classified"
METRICS_CSV    = PROJECT_ROOT / "data" / "urban_growth_metrics.csv"
CHARTS_DIR     = PROJECT_ROOT / "data" / "charts"

CITIES = ["riverside", "phoenix", "las_vegas", "austin"]
YEARS  = [1990, 2000, 2010, 2020]

# Value used for nodata pixels in the classification TIFFs (from Member 3)
NODATA_VALUE = 255

CITY_LABELS = {
    "riverside": "Riverside",
    "phoenix":   "Phoenix",
    "las_vegas": "Las Vegas",
    "austin":    "Austin",
}

CITY_COLORS = {
    "riverside": "#2F6F4E",
    "phoenix":   "#E76F51",
    "las_vegas": "#4C9BE8",
    "austin":    "#7A9E7E",
}

THEME = {
    "background": "#F7F6F2",
    "primary": "#2F6F4E",
    "accent": "#E76F51",
    "secondary": "#4C9BE8",
    "card": "#FFFFFF",
    "text": "#000000",
    "muted": "#000000",
    "border": "#E4E2D9",
}

MAP_CLASS_COLORS = {
    "urban": [64, 64, 64, 255],        # bright red (urban growth highlight)
    "vegetation": [102, 187, 106, 255], # vivid green
    "water": [66, 165, 245, 255],       # bright blue
    "other": [189, 189, 189, 255],      # medium gray
}


# =========================================================
# PAGE CONFIG  (must be the very first Streamlit call)
# =========================================================

st.set_page_config(
    page_title="Urban Expansion Mapping",
    page_icon="U",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GLOBAL CSS — nature-inspired light theme
# =========================================================

st.markdown("""
<style>
/* ── Base backgrounds ──────────────────────────────────────────── */
.stApp                         { background-color: #F7F6F2; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F3F1EA 0%, #ECE8DD 100%);
    border-right: 1px solid #E4E2D9;
}

/* ── Typography ────────────────────────────────────────────────── */
h1, h2, h3 { color: #000000 !important; }
p, li, label { color: #000000; }

/* ── Metric cards ───────────────────────────────────────────────── */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E4E2D9;
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
    margin-bottom: 10px;
    box-shadow: 0 4px 14px rgba(47, 59, 53, 0.06);
}
.metric-value {
    font-size: 2em;
    font-weight: 700;
    color: #000000;
}
.metric-label {
    font-size: 0.82em;
    color: #000000;
    margin-top: 4px;
}

/* ── Highlighted info boxes ─────────────────────────────────────── */
.info-box {
    background: #FFFFFF;
    border-left: 4px solid #4C9BE8;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 6px 0;
    color: #000000;
    font-size: 0.9em;
    border: 1px solid #E4E2D9;
}

/* ── Hero banner ────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(120deg, #F6EFE2 0%, #F7F6F2 45%, #EAF3EE 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 24px;
    border: 1px solid #E4E2D9;
}

/* ── Tabs ───────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab"]          { color: #000000; }
.stTabs [aria-selected="true"]         { color: #000000 !important; }

/* ── Streamlit's native metric widget ──────────────────────────── */
[data-testid="stMetric"] label        { color: #000000 !important; }
[data-testid="stMetricValue"]          { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA LOADERS  (cached so they only run once per session)
# =========================================================

@st.cache_data
def load_metrics() -> pd.DataFrame:
    """
    Load the urban growth metrics CSV produced by Member 4.
    Returns an empty DataFrame when the file does not exist yet.
    """
    if METRICS_CSV.exists():
        return pd.read_csv(METRICS_CSV)
    return pd.DataFrame(
        columns=["city", "year", "urban_area_km2", "growth_km2", "growth_pct", "growth_pct_display"]
    )


@st.cache_data
def load_riverside_training_samples() -> pd.DataFrame:
    """Load and combine all Riverside training CSV rows across available years."""
    frames = []
    csv_files = sorted((PROJECT_ROOT / "data").glob("riverside_*_training_samples.csv"))

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        required = {"year", "subclass", "red", "green", "blue", "nir", "ndvi"}
        if not required.issubset(df.columns):
            continue

        keep = df[["year", "subclass", "red", "green", "blue", "nir", "ndvi"]].copy()
        keep["subclass"] = keep["subclass"].astype(str).str.strip().str.lower()
        for col in ["red", "green", "blue", "nir", "ndvi"]:
            keep[col] = pd.to_numeric(keep[col], errors="coerce")
        keep["year"] = pd.to_numeric(keep["year"], errors="coerce")
        keep = keep.dropna(subset=["year"])
        keep["year"] = keep["year"].astype(int)
        frames.append(keep)

    if not frames:
        return pd.DataFrame(columns=["year", "subclass", "red", "green", "blue", "nir", "ndvi"])

    return pd.concat(frames, ignore_index=True)


@st.cache_data
def load_classification_rgba(city: str, year: int) -> "np.ndarray | None":
    """
    Read a classified GeoTIFF and convert it to an RGBA NumPy array.

        Colour mapping (requested theme)
        -------------------------------
            1  (urban)      → terracotta red
            2  (vegetation) → forest green
            3  (water)      → sky blue
            0  (other)      → gray
            255 (nodata)    → transparent
    """
    data = load_classification_data(city, year)
    if data is None:
        return None

    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Default every valid pixel to "other" gray first.
    rgba[data != NODATA_VALUE] = MAP_CLASS_COLORS["other"]

    # Known class values.
    rgba[data == 1] = MAP_CLASS_COLORS["urban"]
    rgba[data == 2] = MAP_CLASS_COLORS["vegetation"]
    rgba[data == 3] = MAP_CLASS_COLORS["water"]

    # Nodata pixels: fully transparent
    rgba[data == NODATA_VALUE] = [0, 0, 0, 0]

    return rgba


@st.cache_data
def load_classification_data(city: str, year: int) -> "np.ndarray | None":
    """Read a classified GeoTIFF band for a city/year and return raw class values."""
    tif_path = CLASSIFIED_DIR / f"{city}_{year}_classification.tif"
    if not tif_path.exists():
        return None

    with rasterio.open(tif_path) as src:
        return src.read(1)


@st.cache_data
def classification_pil(city: str, year: int, width: int = 640) -> "Image.Image | None":
    """
    Return a PIL Image of the classification map scaled to *width* pixels wide.
    Uses nearest-neighbour resampling to preserve hard classification boundaries.
    """
    rgba = load_classification_rgba(city, year)
    if rgba is None:
        return None
    img    = Image.fromarray(rgba, mode="RGBA")
    aspect = img.height / img.width
    return img.resize((width, int(width * aspect)), Image.NEAREST)


@st.cache_data
def generate_timelapse_gif(city: str) -> "bytes | None":
    """
    Build an animated GIF that cycles through all four years for a given city.
    Each frame is annotated with the city name and year.
    """
    frames = []

    for year in YEARS:
        img = classification_pil(city, year, width=420)

        # Fall back to a blank frame when the file is missing
        if img is None:
            img = Image.new("RGBA", (420, 340), (15, 23, 42, 255))

        # Convert to RGB for drawing
        frame_rgb = img.convert("RGB")

        # Semi-transparent top banner
        banner_overlay = Image.new("RGBA", frame_rgb.size, (0, 0, 0, 0))
        draw_banner    = ImageDraw.Draw(banner_overlay)
        draw_banner.rectangle([0, 0, frame_rgb.width, 52], fill=(15, 23, 42, 210))
        frame_rgba = Image.alpha_composite(frame_rgb.convert("RGBA"), banner_overlay)

        # Year and city label
        draw_label = ImageDraw.Draw(frame_rgba)
        draw_label.text((12, 10), f"{CITY_LABELS[city]}  ·  {year}", fill=(224, 231, 255, 255))
        draw_label.text((12, 30), "  Urban    Non-Urban", fill=(100, 116, 139, 220))

        # Convert to palette mode for GIF compatibility
        frames.append(frame_rgba.convert("P", palette=Image.ADAPTIVE, colors=128))

    if not frames:
        return None

    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,           # loop forever
        duration=1300,    # ms per frame
        optimize=True,
    )
    buf.seek(0)
    return buf.read()


# =========================================================
# CHART HELPERS
# =========================================================

def _dark_fig(figsize=(9, 5)):
    """Return a (fig, ax) pair pre-styled with the nature light theme."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(THEME["background"])
    ax.set_facecolor(THEME["card"])
    ax.tick_params(colors=THEME["muted"])
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["border"])
    ax.grid(True, alpha=0.18, color="#C7C8BC")
    return fig, ax


def chart_area_over_time(df: pd.DataFrame, active_cities: "list[str] | None" = None):
    """
    Line chart — urban area (km²) vs year.
    When *active_cities* is provided, other cities are drawn faded.
    """
    fig, ax = _dark_fig((10, 5))

    for city in CITIES:
        city_data = df[df["city"] == city].sort_values("year")
        highlighted = (active_cities is None) or (city in active_cities)
        alpha  = 1.0 if highlighted else 0.2
        lwidth = 2.5 if highlighted else 1.2

        ax.plot(
            city_data["year"],
            city_data["urban_area_km2"],
            marker="o",
            linewidth=lwidth,
            alpha=alpha,
            label=CITY_LABELS[city],
            color=CITY_COLORS[city],
        )

        # Annotate final data point for highlighted cities
        if highlighted:
            last = city_data.iloc[-1]
            if pd.notna(last["urban_area_km2"]):
                ax.annotate(
                    f"{last['urban_area_km2']:.0f}",
                    xy=(last["year"], last["urban_area_km2"]),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=CITY_COLORS[city],
                    fontweight="bold",
                )

    ax.set_xticks(YEARS)
    ax.set_xlabel("Year",             color=THEME["muted"], fontsize=11)
    ax.set_ylabel("Urban Area (km²)", color=THEME["muted"], fontsize=11)
    ax.set_title("Urban Area Over Time (1990–2020)",
                 color=THEME["text"], fontsize=13, fontweight="bold", pad=12)
    ax.legend(facecolor=THEME["card"], edgecolor=THEME["border"],
              labelcolor=THEME["text"], fontsize=9)
    plt.tight_layout()
    return fig


def chart_city_comparison(df: pd.DataFrame):
    """Bar chart — total urban growth (km²) from 1990 to 2020 per city."""
    comparison = []
    for city in CITIES:
        city_data = df[df["city"] == city].sort_values("year")
        a1990 = city_data.loc[city_data["year"] == 1990, "urban_area_km2"].values
        a2020 = city_data.loc[city_data["year"] == 2020, "urban_area_km2"].values
        if len(a1990) and len(a2020) and pd.notna(a1990[0]) and pd.notna(a2020[0]):
            growth = round(a2020[0] - a1990[0], 2)
            pct    = round(((a2020[0] - a1990[0]) / a1990[0]) * 100, 1) if a1990[0] > 0 else 0
            comparison.append({
                "city":       CITY_LABELS[city],
                "growth_km2": growth,
                "growth_pct": pct,
                "color":      CITY_COLORS[city],
            })
    comp_df = pd.DataFrame(comparison)

    fig, ax = _dark_fig((8, 5))

    if not comp_df.empty:
        bars = ax.bar(
            comp_df["city"],
            comp_df["growth_km2"],
            color=comp_df["color"].tolist(),
            edgecolor="#0f172a",
            linewidth=0.8,
        )
        for bar, (_, row) in zip(bars, comp_df.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(comp_df["growth_km2"]) * 0.02,
                f"+{row['growth_pct']:.0f}%",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#e2e8f0",
            )

    ax.set_title("Total Urban Growth 1990–2020 by City",
                 color=THEME["text"], fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("City",             color=THEME["muted"], fontsize=11)
    ax.set_ylabel("Growth (km²)",     color=THEME["muted"], fontsize=11)
    plt.tight_layout()
    return fig


def chart_city_2020_area(df: pd.DataFrame):
    """Bar chart — urban area (km²) in year 2020 by city."""
    rows = []
    for city in CITIES:
        value = df.loc[(df["city"] == city) & (df["year"] == 2020), "urban_area_km2"].values
        if len(value) and pd.notna(value[0]):
            rows.append({
                "city": CITY_LABELS[city],
                "area_km2": float(value[0]),
                "color": CITY_COLORS[city],
            })

    city_df = pd.DataFrame(rows)
    fig, ax = _dark_fig((8, 5))

    if not city_df.empty:
        bars = ax.bar(
            city_df["city"],
            city_df["area_km2"],
            color=city_df["color"].tolist(),
            edgecolor=THEME["border"],
            linewidth=0.8,
        )
        for bar, (_, row) in zip(bars, city_df.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(city_df["area_km2"]) * 0.02,
                f"{row['area_km2']:.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=THEME["text"],
            )

    ax.set_title("Urban Area in 2020 by City",
                 color=THEME["text"], fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("City", color=THEME["muted"], fontsize=11)
    ax.set_ylabel("Urban Area (km²)", color=THEME["muted"], fontsize=11)
    plt.tight_layout()
    return fig


def chart_single_city_bars(df: pd.DataFrame, city: str):
    """Grouped bar chart — urban area for one city across all four years."""
    city_data  = df[df["city"] == city].sort_values("year")
    year_shades = ["#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8"]

    fig, ax = _dark_fig((7, 4))

    values = [
        float(city_data.loc[city_data["year"] == y, "urban_area_km2"].values[0])
        if len(city_data.loc[city_data["year"] == y]) else 0.0
        for y in YEARS
    ]

    bars = ax.bar([str(y) for y in YEARS], values,
                  color=year_shades, edgecolor="#0f172a", linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + max(values) * 0.02,
                f"{h:.0f}", ha="center", fontsize=9, color="#e2e8f0",
            )

    ax.set_title(f"{CITY_LABELS[city]} - Urban Area by Year",
                 color=THEME["text"], fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Year",             color=THEME["muted"], fontsize=10)
    ax.set_ylabel("Urban Area (km²)", color=THEME["muted"], fontsize=10)
    plt.tight_layout()
    return fig


def chart_riverside_subclass_heatmap(df: pd.DataFrame, subclass: str):
    """Heatmap of mean spectral features by year for one Riverside subclass."""
    features = ["red", "green", "blue", "nir", "ndvi"]
    sub_df = df[df["subclass"] == subclass].copy()

    if sub_df.empty:
        return None

    heat = (
        sub_df.groupby("year")[features]
        .mean()
        .reindex(YEARS)
    )

    if heat.isna().all().all():
        return None

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor(THEME["background"])
    ax.set_facecolor(THEME["card"])

    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="#F2F2F2")
    matrix = np.ma.masked_invalid(heat.values)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels([f.upper() for f in features], color=THEME["muted"], fontsize=9)
    ax.set_yticks(np.arange(len(YEARS)))
    ax.set_yticklabels([str(y) for y in YEARS], color=THEME["muted"], fontsize=9)
    ax.set_title(
        f"Riverside Subclass Heatmap: {subclass.replace('_', ' ').title()}",
        color=THEME["text"],
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Spectral Features", color=THEME["muted"], fontsize=10)
    ax.set_ylabel("Year", color=THEME["muted"], fontsize=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = heat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=THEME["text"])

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=8, colors=THEME["muted"])

    plt.tight_layout()
    return fig


# =========================================================
# PAGE: HOME
# =========================================================

def page_home():
    st.markdown("""
    <div class="hero-banner">
        <h1 style="color:#000000; font-size:2.2em; margin-bottom:4px;">Urban Expansion</h1>
        <p style="color:#000000; margin:0; font-size:1.0em;">CS 224 Final Project</p>
    </div>
    """, unsafe_allow_html=True)

    tab_summary, tab_pipeline = st.tabs([ "Quick-Look Summary", "System Pipeline"])

    with tab_pipeline:
        step_cols = st.columns(5)
        pipeline_steps = [
            ("1", "Landsat Data", "Satellite imagery via Google Earth Engine"),
            ("2", "Preprocessing", "Cloud masking · median composites · band selection"),
            ("3", "Feature Extraction", "Red · Green · Blue · NIR · NDVI"),
            ("4", "Spark ML", "Random Forest classification at scale"),
            ("5", "Analytics", "Urban growth metrics & interactive visualizations"),
        ]
        for col, (icon, title, desc) in zip(step_cols, pipeline_steps):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.8em; color:#000000; font-weight:700;">{icon}</div>
                    <div style="color:#000000; font-weight:600; margin:6px 0; font-size:0.95em;">{title}</div>
                    <div style="color:#000000; font-size:0.78em;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab_summary:
        df = load_metrics()
        if not df.empty:
            summary_cols = st.columns(4)
            for col, city in zip(summary_cols, CITIES):
                city_data = df[df["city"] == city].sort_values("year")
                a1990 = city_data.loc[city_data["year"] == 1990, "urban_area_km2"].values
                a2010 = city_data.loc[city_data["year"] == 2010, "urban_area_km2"].values
                with col:
                    if len(a1990) and len(a2010) and pd.notna(a1990[0]) and pd.notna(a2010[0]):
                        pct = round(((a2010[0] - a1990[0]) / a1990[0]) * 100, 1)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="color:#000000; font-size:1.1em; font-weight:700;">
                                {CITY_LABELS[city]}
                            </div>
                            <div class="metric-value" style="color:#000000;">{pct:+.0f}%</div>
                            <div class="metric-label">urban growth (1990–2010)</div>
                            <div style="color:#000000; font-size:0.78em; margin-top:6px;">
                                {a2010[0]:.0f} km² in 2010
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="color:#000000; font-weight:700;">{CITY_LABELS[city]}</div>
                            <div style="color:#000000; font-size:0.85em; margin-top:8px;">Data unavailable —<br>run Member 4 script first</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info(
                "Metrics data not found. Run `python scripts/member4/urban_growth_metrics.py` "
                "from the project root to generate `data/urban_growth_metrics.csv`."
            )


# =========================================================
# PAGE: CLASSIFICATION MAPS
# =========================================================

def page_classification_maps():
    st.title("Main Map")
    st.markdown("Large central map showing the classification raster with urban areas highlighted.")

    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        selected_city = st.selectbox(
            "City", CITIES, format_func=lambda c: CITY_LABELS[c], key="map_city"
        )
    with sel_col2:
        selected_year = st.selectbox("Year", YEARS, key="map_year")

    map_col, info_col = st.columns([3, 1])

    with map_col:
        img = classification_pil(selected_city, selected_year, width=900)
        if img is not None:
            st.image(
                img,
                caption=f"{CITY_LABELS[selected_city]} {selected_year}  —  Classification Map",
                use_container_width=True,
            )
        else:
            st.warning(
                f"Classification map not found for {CITY_LABELS[selected_city]} {selected_year}.\n\n"
                "Run `scripts/member3/randomForest_toComposites.py` to generate it."
            )

    with info_col:
        df = load_metrics()
        row = df[(df["city"] == selected_city) & (df["year"] == selected_year)]

        st.markdown("### Metrics")
        if not row.empty and pd.notna(row.iloc[0]["urban_area_km2"]):
            r    = row.iloc[0]
            area = r["urban_area_km2"]
            gpct = r["growth_pct"]

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Urban Area</div>
                <div class="metric-value">{area:.1f} km²</div>
            </div>
            """, unsafe_allow_html=True)

            if pd.notna(gpct):
                arrow = "▲" if gpct > 0 else "▼"
                clr   = THEME["primary"] if gpct > 0 else THEME["accent"]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Growth from previous decade</div>
                    <div class="metric-value" style="color:{clr};">{arrow} {abs(gpct):.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="info-box">Baseline year — no prior decade to compare.</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="info-box">Run the metrics script to see data here.</div>',
                unsafe_allow_html=True,
            )

# =========================================================
# PAGE: SIDE-BY-SIDE COMPARISON
# =========================================================

def page_side_by_side():
    st.title("Side-by-Side Comparison")
    st.markdown("Directly compare classification maps across years or cities.")

    tab_years, tab_cities = st.tabs(["Compare Years — Same City", "Compare Cities — Same Year"])

    # ── Tab 1: same city, different years ────────────────────────────────────
    with tab_years:
        city_sbs = st.selectbox(
            "City", CITIES, format_func=lambda c: CITY_LABELS[c], key="sbs_city"
        )
        yr_col1, yr_col2 = st.columns(2)
        with yr_col1:
            year_a = st.selectbox("Year A", YEARS, index=0, key="sbs_yr_a")
        with yr_col2:
            year_b = st.selectbox("Year B", YEARS, index=3, key="sbs_yr_b")

        map_a_col, map_b_col = st.columns(2)
        img_a = classification_pil(city_sbs, year_a, width=520)
        img_b = classification_pil(city_sbs, year_b, width=520)

        with map_a_col:
            if img_a:
                st.image(img_a, caption=f"{CITY_LABELS[city_sbs]} {year_a}", use_container_width=True)
            else:
                st.warning("Map not available.")
        with map_b_col:
            if img_b:
                st.image(img_b, caption=f"{CITY_LABELS[city_sbs]} {year_b}", use_container_width=True)
            else:
                st.warning("Map not available.")

        # Delta metrics below the maps
        df = load_metrics()
        val_a = df[(df["city"] == city_sbs) & (df["year"] == year_a)]["urban_area_km2"].values
        val_b = df[(df["city"] == city_sbs) & (df["year"] == year_b)]["urban_area_km2"].values

        if len(val_a) and len(val_b) and pd.notna(val_a[0]) and pd.notna(val_b[0]):
            delta    = val_b[0] - val_a[0]
            delta_pct = (delta / val_a[0]) * 100 if val_a[0] > 0 else 0
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Urban Area — {year_a}", f"{val_a[0]:.1f} km²")
            m2.metric(f"Urban Area — {year_b}", f"{val_b[0]:.1f} km²",
                      delta=f"{delta:+.1f} km²")
            m3.metric("% Change", f"{delta_pct:+.1f}%")

    # ── Tab 2: same year, different cities ────────────────────────────────────
    with tab_cities:
        year_sbs = st.selectbox("Year", YEARS, index=3, key="sbs_year_only")
        ct_col1, ct_col2 = st.columns(2)
        with ct_col1:
            city_a = st.selectbox(
                "City A", CITIES, index=0,
                format_func=lambda c: CITY_LABELS[c], key="sbs_city_a"
            )
        with ct_col2:
            city_b = st.selectbox(
                "City B", CITIES, index=1,
                format_func=lambda c: CITY_LABELS[c], key="sbs_city_b"
            )

        ca_col, cb_col = st.columns(2)
        img_ca = classification_pil(city_a, year_sbs, width=520)
        img_cb = classification_pil(city_b, year_sbs, width=520)

        with ca_col:
            if img_ca:
                st.image(img_ca, caption=f"{CITY_LABELS[city_a]} {year_sbs}", use_container_width=True)
            else:
                st.warning("Map not available.")
        with cb_col:
            if img_cb:
                st.image(img_cb, caption=f"{CITY_LABELS[city_b]} {year_sbs}", use_container_width=True)
            else:
                st.warning("Map not available.")

        # Side-by-side urban area metrics for the two cities
        df = load_metrics()
        va = df[(df["city"] == city_a) & (df["year"] == year_sbs)]["urban_area_km2"].values
        vb = df[(df["city"] == city_b) & (df["year"] == year_sbs)]["urban_area_km2"].values
        if (len(va) and len(vb) and pd.notna(va[0]) and pd.notna(vb[0])):
            diff  = va[0] - vb[0]
            m1, m2, m3 = st.columns(3)
            m1.metric(CITY_LABELS[city_a],      f"{va[0]:.1f} km²")
            m2.metric(CITY_LABELS[city_b],      f"{vb[0]:.1f} km²")
            m3.metric("Difference",             f"{abs(diff):.1f} km²",
                      delta=f"{diff:+.1f} km² ({CITY_LABELS[city_a]} vs {CITY_LABELS[city_b]})")


# =========================================================
# PAGE: ANIMATED TIME-LAPSE
# =========================================================

def page_timelapse():
    st.title("Animated Time-Lapse")
    st.markdown(
        "Watch urban areas expand from **1990** to **2020**. "
        "Use the animated GIF tab for a looping animation, or "
        "drag the slider to step through each decade manually."
    )

    # City toggle — supports the "toggle between cities" requirement
    tl_city = st.selectbox(
        "Select City", CITIES, format_func=lambda c: CITY_LABELS[c], key="tl_city"
    )

    tab_gif, tab_slider = st.tabs(["Animated GIF", "Step Through Manually"])

    with tab_gif:
        with st.spinner(f"Generating time-lapse for {CITY_LABELS[tl_city]}…"):
            gif_bytes = generate_timelapse_gif(tl_city)

        if gif_bytes:
            # Centre the GIF
            _, gif_col, _ = st.columns([1, 2, 1])
            with gif_col:
                st.image(
                    gif_bytes,
                    caption=f"{CITY_LABELS[tl_city]} — Urban Growth 1990–2020",
                    use_container_width=True,
                )
            st.download_button(
                label="Download GIF",
                data=gif_bytes,
                file_name=f"{tl_city}_urban_timelapse.gif",
                mime="image/gif",
            )
        else:
            st.warning("Could not generate time-lapse — classification files may be missing.")

    with tab_slider:
        idx  = st.slider("Decade", min_value=0, max_value=3, value=0,
                          format="%d", help="Drag left/right to change the decade")
        year = YEARS[idx]
        st.caption(f"**{CITY_LABELS[tl_city]}  ·  {year}**")

        _, slider_col, _ = st.columns([1, 2, 1])
        with slider_col:
            slide_img = classification_pil(tl_city, year, width=500)
            if slide_img:
                st.image(slide_img, use_container_width=True)
            else:
                st.warning("Map not found for this city / year.")

        # Visual decade timeline
        st.markdown("---")
        tl_cols = st.columns(4)
        for tc, y in zip(tl_cols, YEARS):
            indicator = "Current" if y == year else "-"
            tc.markdown(
                f"<div style='text-align:center; color:#6F7E73;'>"
                f"{indicator}<br><small>{y}</small></div>",
                unsafe_allow_html=True,
            )


# =========================================================
# PAGE: URBAN AREA vs TIME  (with city toggles)
# =========================================================

def page_urban_growth_trends():
    st.title("Urban Growth Trends")
    st.markdown("Urban Area Growth Over Time and city comparison charts.")

    df = load_metrics()
    if df.empty:
        st.warning(
            "No metrics data found. "
            "Run `python scripts/member4/urban_growth_metrics.py` first."
        )
        return

    # City toggle checkboxes — "toggle between cities" feature
    st.markdown("### Toggle Cities")
    toggle_cols = st.columns(4)
    city_toggles = {}
    for i, city in enumerate(CITIES):
        with toggle_cols[i]:
            city_toggles[city] = st.checkbox(
                CITY_LABELS[city], value=True, key=f"toggle_{city}"
            )

    active = [c for c, on in city_toggles.items() if on]
    if not active:
        st.warning("Select at least one city to display.")
        return

    # Main line chart
    st.markdown("### Urban Area Growth Over Time")
    fig = chart_area_over_time(df, active_cities=active)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### City Comparison Chart")
    st.caption("Urban Area in 2020 across Riverside, Phoenix, Las Vegas, and Austin")
    fig = chart_city_2020_area(df)
    st.pyplot(fig)
    plt.close(fig)

    # Growth rates table
    st.markdown("### Growth Rates Table")
    display_cols   = ["city", "year", "urban_area_km2", "growth_km2", "growth_pct_display"]
    available_cols = [c for c in display_cols if c in df.columns]
    table_df       = df[df["city"].isin(active)][available_cols].copy()
    table_df["city"] = table_df["city"].map(CITY_LABELS)
    table_df.columns = [
        c.replace("_km2", " (km²)").replace("_pct_display", " %")
         .replace("_", " ").title()
        for c in table_df.columns
    ]
    st.dataframe(table_df, use_container_width=True)


# =========================================================
# PAGE: RIVERSIDE LANDCOVER TRENDS
# =========================================================

def page_riverside_landcover():
    st.title("Riverside Subclass Heatmaps")
    st.caption("Each heatmap shows mean Red/Green/Blue/NIR/NDVI values by year for one subclass.")

    subclass_df = load_riverside_training_samples()
    if subclass_df.empty:
        st.warning("No Riverside training samples available to generate subclass heatmaps.")
    else:
        preferred_order = ["urban", "vegetation", "farmland", "bare_soil", "water"]
        available = sorted(subclass_df["subclass"].dropna().unique().tolist())
        subclass_order = [s for s in preferred_order if s in available]
        subclass_order += [s for s in available if s not in subclass_order]

        heat_tabs = st.tabs([s.replace("_", " ").title() for s in subclass_order])
        for tab, subclass in zip(heat_tabs, subclass_order):
            with tab:
                fig = chart_riverside_subclass_heatmap(subclass_df, subclass)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info(f"No valid data available for subclass: {subclass}.")


# =========================================================
# PAGE: CITY COMPARISON
# =========================================================

def page_city_comparison():
    st.title("City Comparison")
    st.markdown("Compare urban growth patterns across all four cities side by side.")

    df = load_metrics()
    if df.empty:
        st.warning(
            "No metrics data found. "
            "Run `python scripts/member4/urban_growth_metrics.py` first."
        )
        return

    # Two main charts side by side
    chart_l, chart_r = st.columns(2)
    with chart_l:
        st.markdown("##### Total Growth 1990–2020")
        fig = chart_city_comparison(df)
        st.pyplot(fig)
        plt.close(fig)

    with chart_r:
        st.markdown("##### Urban Area Over Time")
        fig = chart_area_over_time(df)
        st.pyplot(fig)
        plt.close(fig)

    # Analysis highlights
    st.markdown("### Analysis Highlights")
    summary_data = []
    for city in CITIES:
        city_data = df[df["city"] == city].sort_values("year")
        a1990 = city_data.loc[city_data["year"] == 1990, "urban_area_km2"].values
        a2020 = city_data.loc[city_data["year"] == 2020, "urban_area_km2"].values
        if len(a1990) and len(a2020) and pd.notna(a1990[0]) and pd.notna(a2020[0]):
            growth = a2020[0] - a1990[0]
            pct    = ((a2020[0] - a1990[0]) / a1990[0]) * 100 if a1990[0] > 0 else 0
            summary_data.append({"city": city, "growth_km2": growth, "growth_pct": pct})

    if summary_data:
        sum_df   = pd.DataFrame(summary_data)
        fastest  = sum_df.loc[sum_df["growth_pct"].idxmax()]
        largest  = sum_df.loc[sum_df["growth_km2"].idxmax()]

        hi_l, hi_r = st.columns(2)
        with hi_l:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#fbbf24; font-size:1em;">Fastest % Growth (1990–2020)</div>
                <div class="metric-value" style="color:#fbbf24;">{CITY_LABELS[fastest['city']]}</div>
                <div class="metric-label">+{fastest['growth_pct']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with hi_r:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#f87171; font-size:1em;">Largest Absolute Growth</div>
                <div class="metric-value" style="color:#f87171;">{CITY_LABELS[largest['city']]}</div>
                <div class="metric-label">+{largest['growth_km2']:.1f} km²</div>
            </div>
            """, unsafe_allow_html=True)

    # Decade-by-decade expansion totals
    st.markdown("### Decade-by-Decade Expansion (All Cities Combined)")
    decade_cols = st.columns(3)
    for dcol, (yr_a, yr_b) in zip(decade_cols, [(1990, 2000), (2000, 2010), (2010, 2020)]):
        total = 0.0
        for city in CITIES:
            s = df[(df["city"] == city) & (df["year"] == yr_a)]["urban_area_km2"].values
            e = df[(df["city"] == city) & (df["year"] == yr_b)]["urban_area_km2"].values
            if len(s) and len(e) and pd.notna(s[0]) and pd.notna(e[0]):
                total += e[0] - s[0]
        with dcol:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{yr_a} – {yr_b}</div>
                <div class="metric-value">+{total:.0f}</div>
                <div class="metric-label">km² total</div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

PAGES = {
    "Home":                    page_home,
    "Riverside Landcover":     page_riverside_landcover,
    "Main Map":                page_classification_maps,
    "Side-by-Side":            page_side_by_side,
    "Urban Growth Trends":     page_urban_growth_trends,
    "Time-Lapse":              page_timelapse,
    "City Comparison":         page_city_comparison,
}

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:14px 0 6px 0;">
        <div style="color:#000000; font-weight:700; font-size:1.05em;">Urban Expansion</div>
        <div style="color:#000000; font-size:0.78em; margin-top:2px;">CS 224 Final Project</div>
    </div>
    <hr style="border-color:#E4E2D9; margin:10px 0 14px 0;">
    """, unsafe_allow_html=True)

    page_name = st.radio(
        "Navigate", list(PAGES.keys()), label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#E4E2D9; margin:14px 0 10px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#000000; font-size:0.75em; line-height:1.8;">
        <strong style="color:#000000;">Data</strong><br>
        Landsat 5 / 7 / 8<br>
        Google Earth Engine<br><br>
        <strong style="color:#000000;">Processing</strong><br>
        Apache Spark · Random Forest<br><br>
        <strong style="color:#000000;">Cities</strong><br>
        Riverside · Phoenix<br>
        Las Vegas · Austin<br><br>
        <strong style="color:#000000;">Time Range</strong><br>
        1990 · 2000 · 2010 · 2020
    </div>
    """, unsafe_allow_html=True)

# Render the selected page
PAGES[page_name]()
