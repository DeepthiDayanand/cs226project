#Import the landsatHelpers to be used for the Composites
from landsatHelpers import (
    initialize_earth_engine,
    get_composite,
    get_image_count,
    create_export_task,
)

# Test exporting script, before we run it for all cities and year. Test on Riverside, 1990
# Check: GEE Initialization, Region Geometry, Landsat (5,7,8), Composite Bands, Exporting Works

PROJECT_ID = "cs224project-490217"
CITY_NAME = "riverside"
YEAR = 1990


def main():
    # Initialize Earth Engine.
    initialize_earth_engine(PROJECT_ID, authenticate=False)

    # Import Regions from other file
    from regions import REGIONS

    region = REGIONS[CITY_NAME]

    # Count satellite images for each city and corresponding year
    image_count = get_image_count(region, YEAR)
    print(f"Image count for {CITY_NAME} {YEAR}: {image_count}")

    # Build composite image for all images in that year
    composite = get_composite(region, YEAR)

    # Output bands
    print("Band names:", composite.bandNames().getInfo())

    # Start export task
    task = create_export_task(composite, region, CITY_NAME, YEAR)
    task.start()

    print(f"Export task started for {CITY_NAME}_{YEAR}_composite")
    print(task.status())
    print("Check Google Drive folder: urban_expansion_exports")


if __name__ == "__main__":
    main()