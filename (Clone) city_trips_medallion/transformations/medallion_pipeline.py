from pyspark import pipelines as dp
from pyspark.sql import functions as F

RAW = "/Volumes/prod/landing/raw"

# ---------------------------------------------------------
# BRONZE: raw data as-is, plus ingestion metadata
# ---------------------------------------------------------
TRIPS_SCHEMA = """
    trip_id STRING, 
    date STRING, 
    city_id STRING, 
    passenger_type STRING,
    distance_travelled_km STRING, 
    fare_amount STRING,
    passenger_rating STRING, 
    driver_rating STRING
"""

@dp.table(name="bronze_trips", comment="Raw monthly trips files")
def bronze_trips():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .schema(TRIPS_SCHEMA)  # positional, so headers with spaces don't matter
        .load(f"{RAW}/trips/")
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
    )

@dp.materialized_view(name="bronze_city", comment="Raw city lookup")
def bronze_city():
    return (
        spark.read.format("csv")
        .option("header", "true")
        .schema("city_id STRING, city_name STRING")
        .load(f"{RAW}/city/")
        .withColumn("_ingested_at", F.current_timestamp())
    )

# ---------------------------------------------------------
# SILVER: typed, cleaned, validated
# ---------------------------------------------------------
@dp.table(name="silver_trips", comment="Cleaned and typed trips")
@dp.expect_or_drop("valid_trip_id", "trip_id IS NOT NULL")
@dp.expect_or_drop("valid_city_id", "city_id IS NOT NULL")
@dp.expect_or_drop("valid_date", "trip_date IS NOT NULL")
@dp.expect_or_drop("valid_fare", "fare_amount >= 0")
@dp.expect_or_drop("valid_distance", "distance_travelled > 0")
@dp.expect("valid_passenger_rating", "passenger_rating BETWEEN 1 AND 10")
@dp.expect("valid_driver_rating", "driver_rating BETWEEN 1 AND 10")
def silver_trips():
    return (
        spark.readStream.table("bronze_trips")
        .select(
            F.trim("trip_id").alias("trip_id"),
            F.to_date("date", "yyyy-MM-dd").alias("trip_date"),  # adjust format if needed
            F.trim("city_id").alias("city_id"),
            F.initcap(F.trim("passenger_type")).alias("passenger_type"),
            F.col("distance_travelled_km").cast("double").alias("distance_travelled"),
            F.col("fare_amount").cast("double").alias("fare_amount"),
            F.col("passenger_rating").cast("double").alias("passenger_rating"),
            F.col("driver_rating").cast("double").alias("driver_rating"),
            "_source_file", "_ingested_at",
        )
        .dropDuplicates(["trip_id"])
    )

@dp.materialized_view(name="silver_city", comment="Cleaned city dimension")
# @dp.expect_or_drop("valid_city_id", "city_id IS NOT NULL")
def silver_city():
    return (
        spark.read.table("bronze_city")
        .select(F.trim("city_id").alias("city_id"),
                F.initcap(F.trim("city_name")).alias("city_name"))
        .dropDuplicates(["city_id"])
    )

# ---------------------------------------------------------
# GOLD: business-ready aggregates
# ---------------------------------------------------------
@dp.materialized_view(name="gold_city_monthly_summary",
                      comment="Monthly KPIs per city")
def gold_city_monthly_summary():
    t = spark.read.table("silver_trips")
    c = spark.read.table("silver_city")
    return (
        t.join(c, "city_id", "left")
        .groupBy("city_id", "city_name", F.date_trunc("month", "trip_date").alias("month"))
        .agg(
            F.count("trip_id").alias("total_trips"),
            F.round(F.sum("fare_amount"), 2).alias("total_revenue"),
            F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
            F.round(F.sum("distance_travelled"), 2).alias("total_distance"),
            F.round(F.avg("passenger_rating"), 2).alias("avg_passenger_rating"),
            F.round(F.avg("driver_rating"), 2).alias("avg_driver_rating"),
        )
    )

@dp.materialized_view(name="gold_passenger_type_summary",
                      comment="Trips and revenue by city and passenger type")
def gold_passenger_type_summary():
    t = spark.read.table("silver_trips")
    c = spark.read.table("silver_city")
    return (
        t.join(c, "city_id", "left")
        .groupBy("city_name", "passenger_type")
        .agg(F.count("trip_id").alias("total_trips"),
             F.round(F.sum("fare_amount"), 2).alias("total_revenue"),
             F.round(F.avg("distance_travelled"), 2).alias("avg_distance"))
    )