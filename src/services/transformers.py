import time
from pyspark.sql.functions import col, explode, split, arrays_zip, from_unixtime, year, month, dayofmonth, to_timestamp, regexp_replace

class BronzeToSilverTransformer:
    def __init__(self, spark):
        self.spark = spark

    def log_step(self, message):
        """ Logs the current step with a timestamp to track progress. """
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def transform_treino(self, input_path: str, output_path: str):
        self.log_step("Starting 'Treino' transformation...")

        file_path = f"{input_path}/files/treino/"
        self.log_step("Reading CSV files...")
        df = self.spark.read.option("header", "true").csv(file_path).repartition(8)
        self.log_step("Finished reading CSV files.")

        cols_to_split = [
            "history", "timestampHistory", "numberOfClicksHistory", 
            "timeOnPageHistory", "scrollPercentageHistory", 
            "pageVisitsCountHistory"
        ]

        self.log_step("Splitting and transforming columns...")
        for col_name in cols_to_split:
            df = df.withColumn(col_name, split(col(col_name), ",\\s*"))

        self.log_step("Zipping and exploding columns...")
        df = df.withColumn("zipped", arrays_zip(*[col(c) for c in cols_to_split]))
        df_exploded = df.withColumn("exploded", explode(col("zipped")))

        self.log_step("Normalizing columns...")
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

        self.log_step("Adding partition columns...")
        df_partitioned = df_normalized \
            .withColumn("year", year(col("timestampHistory"))) \
            .withColumn("month", month(col("timestampHistory"))) \
            .withColumn("day", dayofmonth(col("timestampHistory")))

        self.log_step("Writing partitioned Parquet files...")
        df_partitioned.coalesce(1).write \
            .mode("overwrite") \
            .option("compression", "snappy") \
            .partitionBy("year", "month", "day") \
            .parquet(output_path)

        self.log_step("'Treino' transformation completed and data saved.")

    def transform_itens(self, input_path: str, output_path: str):
        self.log_step("Starting 'Itens' transformation...")

        file_path = f"{input_path}/itens/itens/"
        self.log_step("Reading CSV files...")
        df = self.spark.read \
            .option("header", "true") \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .option("multiLine", "true") \
            .option("inferSchema", "true") \
            .csv(file_path).repartition(10)
        self.log_step("Finished reading CSV files.")

        self.log_step("Cleaning up timestamp columns...")
        df = df.withColumn("issued", regexp_replace(col("issued"), r"\+00:00", "")) \
               .withColumn("modified", regexp_replace(col("modified"), r"\+00:00", ""))

        self.log_step("Converting strings to timestamp columns...")
        df = df.withColumn("issued", to_timestamp(col("issued"), "yyyy-MM-dd HH:mm:ss")) \
               .withColumn("modified", to_timestamp(col("modified"), "yyyy-MM-dd HH:mm:ss"))

        self.log_step("Dropping unused columns...")
        df = df.drop("url")

        self.log_step("Adding partition columns...")
        df = df.withColumn("year", year(col("issued"))) \
               .withColumn("month", month(col("issued"))) \
               .withColumn("day", dayofmonth(col("issued")))

        self.log_step("Writing partitioned Parquet files...")
        df.coalesce(1).write \
            .mode("overwrite") \
            .option("compression", "snappy") \
            .partitionBy("year", "month", "day") \
            .parquet(output_path)

        self.log_step("'Itens' transformation completed and data saved.")
