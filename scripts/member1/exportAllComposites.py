#Import the landsatHelpers to be used for the Composites
from landsatHelpers import (
    initialize_earth_engine,
    get_composite,
    get_image_count,
    create_export_task,
)


# We are trying to export all of the cities time period years composites. 
# Script goes through (All 4 of our cities, for the 4 year we are comparing )

# Initalize ProjectID (From GEE)
PROJECT_ID = "cs224project-490217"
YEARS = [1990, 2000, 2010, 2020]


def main():
    # Initialize Earth Engine
    initialize_earth_engine(PROJECT_ID, authenticate=False)

    # Import all cities
    from regions import REGIONS

    for city_name, region in REGIONS.items():
        for year in YEARS:
            print("=" * 60)
            print(f"Preparing export for {city_name} {year}")

            # Count satellite images for each city and corresponding year
            image_count = get_image_count(region, year)
            print(f"Image count: {image_count}")

            # No images check
            if image_count == 0:
                print(f"Skipping {city_name} {year}: no images found")
                continue

            # Build composite image for all images in that year
            composite = get_composite(region, year)

            # Output bands
            bands = composite.bandNames().getInfo()
            print(f"Bands: {bands}")

            # Start export task
            task = create_export_task(composite, region, city_name, year)
            task.start()

            print(f"Started export task: {city_name}_{year}_composite")

    print("=" * 60)
    print("All export tasks have been started.")
    print("Check Google Drive folder: urban_expansion_exports")

if __name__ == "__main__":
    main()