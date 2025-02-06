from pyspark.sql.functions import col, explode, split, arrays_zip, from_unixtime, year, month, dayofmonth, to_timestamp, regexp_replace

class BronzeToSilverTransformer:
    def __init__(self, spark):
        self.spark = spark

    def transform_treino(self, input_path: str, output_path: str):
        print("Starting 'Treino' transformation...")

        file_path = f"{input_path}/files/treino/"
        df = self.spark.read.option("header", "true").csv(file_path)

        # Splitting and transforming columns
        cols_to_split = [
            "history", "timestampHistory", "numberOfClicksHistory", 
            "timeOnPageHistory", "scrollPercentageHistory", 
            "pageVisitsCountHistory"
        ]

        for col_name in cols_to_split:
            df = df.withColumn(col_name, split(col(col_name), ",\\s*"))

        df = df.withColumn("zipped", arrays_zip(*[col(c) for c in cols_to_split]))
        df_exploded = df.withColumn("exploded", explode(col("zipped")))

        df_normalized = df_exploded.select(
            col("userId"),
            col("userType"),
            col("exploded.history").alias("history"),
            from_unixtime(col("exploded.timestampHistory").cast("long") / 1000, "yyyy-MM-dd HH:mm:ss").alias("timestampHistory"),
            col("exploded.numberOfClicksHistory").cast("int").alias("numberOfClicksHistory"),
            col("exploded.timeOnPageHistory").cast("int").alias("timeOnPageHistory"),
            col("exploded.scrollPercentageHistory").cast("float").alias("scrollPercentageHistory"),
            col("exploded.pageVisitsCountHistory").alias("pageVisitsCountHistory")
        )

        # Partitioning by year, month, and day
        df_partitioned = df_normalized \
            .withColumn("year", year(col("timestampHistory"))) \
            .withColumn("month", month(col("timestampHistory"))) \
            .withColumn("day", dayofmonth(col("timestampHistory")))

        df_partitioned.write \
            .mode("overwrite") \
            .option("compression", "snappy") \
            .partitionBy("year", "month", "day") \
            .parquet(output_path)

        print("'Treino' transformation completed and data saved.")

    def transform_itens(self, input_path: str, output_path: str):
        print("Starting 'Itens' transformation...")

        file_path = f"{input_path}/itens/itens/"
        df = self.spark.read \
            .option("header", "true") \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .option("multiLine", "true") \
            .option("inferSchema", "true") \
            .csv(file_path)

        df = df.withColumn("issued", regexp_replace(col("issued"), r"\+00:00", "")) \
               .withColumn("modified", regexp_replace(col("modified"), r"\+00:00", ""))

        df = df.withColumn("issued", to_timestamp(col("issued"), "yyyy-MM-dd HH:mm:ss")) \
               .withColumn("modified", to_timestamp(col("modified"), "yyyy-MM-dd HH:mm:ss"))

        df = df.drop("url")

        df = df.withColumn("year", year(col("issued"))) \
               .withColumn("month", month(col("issued"))) \
               .withColumn("day", dayofmonth(col("issued")))

        df.write \
            .mode("overwrite") \
            .option("compression", "snappy") \
            .partitionBy("year", "month", "day") \
            .parquet(output_path)

        print("'Itens' transformation completed and data saved.")