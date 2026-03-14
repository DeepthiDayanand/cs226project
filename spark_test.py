from pyspark.sql import SparkSession

def main():
    # Start a Spark session
    spark = SparkSession.builder \
        .appName("UrbanExpansionDevTest") \
        .master("local[*]") \
        .getOrCreate()

    # Sample data
    data = [
        ("Riverside", 1990, 45.2),
        ("Phoenix", 2000, 88.7),
        ("Las Vegas", 2010, 73.4),
        ("Austin", 2020, 91.1)
    ]

    columns = ["city", "year", "urban_area_km2"]

    # Create Spark DataFrame
    df = spark.createDataFrame(data, columns)

    print("\nSchema:")
    df.printSchema()

    print("\nData:")
    df.show()

    print("\nCities with urban area > 70 km²:")
    df.filter(df.urban_area_km2 > 70).show()

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()