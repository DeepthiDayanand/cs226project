import ee

# Helper Function are needed, other files utilize these helpers

# Initialize Earth Engine
def initialize_earth_engine(project_id: str, authenticate: bool = False) -> None:
    if authenticate:
        ee.Authenticate()

    ee.Initialize(project=project_id)

# Convering the Landsat Surface Reflectance into what is actual Reflectance
def apply_scale_factors(image: ee.Image) -> ee.Image:
    # Replace original bands with the scaled ones
    optical = image.select("SR_B.*").multiply(2.75e-05).add(-0.2)
    return image.addBands(optical, overwrite=True)

# NO clouds in our images, so we have no contaiminated pixels
# Using cloud masking 
def mask_landsat_clouds(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    
    # cloud masking
    cloud_shadow_bit = 1 << 4
    cloud_bit = 1 << 3

    mask = qa.bitwiseAnd(cloud_shadow_bit).eq(0).And(
        qa.bitwiseAnd(cloud_bit).eq(0)
    )

    return image.updateMask(mask)

# Coordinating the Years With its Correct Landsat Model 
# - 1990, 2000 -> Landsat 5
# - 2010 -> Landsat 7
# - 2020 -> Landsat 8
def get_collection_id(year: int) -> str:
    if year in [1990, 2000]:
        return "LANDSAT/LT05/C02/T1_L2"

    if year == 2010:
        return "LANDSAT/LE07/C02/T1_L2"

    if year == 2020:
        return "LANDSAT/LC08/C02/T1_L2"

    raise ValueError(
        f"Unsupported year: {year}. Use one of 1990, 2000, 2010, 2020."
    )

# Date Window For the Target Year (Each Composite Year has Corresponding Date Window) (1990 -> 1989-01-01 to 1991-12-31)
def get_date_window(year: int) -> tuple[str, str]:
    start = f"{year - 1}-01-01"
    end = f"{year + 1}-12-31"
    return start, end

# Ensuring the Band Are Standardized For the Landsat Sensors (red, green, blue, nir)
# Quick Logical Help (more for us)
# Landsat 5 and Landsat 7: SR_B1 = blue, SR_B2 = green, SR_B3 = red, SR_B4 = nir
# Landsat 8: SR_B2 = blue, SR_B3 = green, SR_B4 = red, SR_B5 = nir
def rename_bands(image: ee.Image, year: int) -> ee.Image:
    if year in [1990, 2000, 2010]:
        return image.select(
            ["SR_B3", "SR_B2", "SR_B1", "SR_B4"],
            ["red", "green", "blue", "nir"],
        )

    if year == 2020:
        return image.select(
            ["SR_B4", "SR_B3", "SR_B2", "SR_B5"],
            ["red", "green", "blue", "nir"],
        )

    raise ValueError(f"Unsupported year for renaming: {year}")

# For Each Composite Process Is (choose collection, filter by region and date window, scale, mask cloud, make median composite, rename bands)
def get_composite(region: ee.Geometry, year: int) -> ee.Image:
    collection_id = get_collection_id(year)
    start_date, end_date = get_date_window(year)

    composite = (
        ee.ImageCollection(collection_id)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .map(apply_scale_factors)
        .map(mask_landsat_clouds)
        .median()
    )

    composite = rename_bands(composite, year)

    return composite


# For our return knowledge, how many images for a city and the corresponding year. Four Our Reference
def get_image_count(region: ee.Geometry, year: int) -> int:
    collection_id = get_collection_id(year)
    start_date, end_date = get_date_window(year)

    count = (
        ee.ImageCollection(collection_id)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .size()
    )

    return count.getInfo()


def create_export_task(
    image: ee.Image,
    region: ee.Geometry,
    city_name: str,
    year: int,
    folder: str = "urban_expansion_exports",
) -> ee.batch.Task:
    """
    Google Drive export task for a composite image.
    """
    file_prefix = f"{city_name}_{year}_composite"

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=file_prefix,
        folder=folder,
        fileNamePrefix=file_prefix,
        region=region,
        scale=30,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )

    return task