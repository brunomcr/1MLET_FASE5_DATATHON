from utils.logger import logger
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, split, arrays_zip, from_unixtime, to_timestamp, regexp_replace,
    hour, dayofweek, month, current_timestamp, lag, avg, min, max, log1p, when, udf
)
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf


class BronzeToSilverTransformer:
    def __init__(self, spark):
        self.spark = spark

    def log_step(self, message):
        """ Logs the current step with a timestamp to track progress. """
        logger.info(message)

    def transform_treino(self, input_path: str, output_path: str):
        self.log_step("Starting 'Treino' transformation...")

        file_path = f"{input_path}/files/treino/"
        self.log_step(f"Reading CSV files from {file_path}...")
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
            (col("exploded.timestampHistory").cast("long") / 1000).alias("timestampHistory"),
            col("exploded.numberOfClicksHistory").cast("int").alias("numberOfClicksHistory"),
            col("exploded.timeOnPageHistory").cast("int").alias("timeOnPageHistory"),
            col("exploded.scrollPercentageHistory").cast("float").alias("scrollPercentageHistory"),
            col("exploded.pageVisitsCountHistory").cast("int").alias("pageVisitsCountHistory")
        )

        self.log_step("Adding temporal features...")
        df_temporal = df_normalized \
            .withColumn("hour", hour(from_unixtime(col("timestampHistory")))) \
            .withColumn("dayofweek", dayofweek(from_unixtime(col("timestampHistory")))) \
            .withColumn("month", month(from_unixtime(col("timestampHistory"))))

        window_spec = Window.partitionBy("userId").orderBy("timestampHistory")
        df_temporal = df_temporal \
            .withColumn("time_since_last_interaction",
                        col("timestampHistory") - lag("timestampHistory").over(window_spec)) \
            .fillna(0, subset=["time_since_last_interaction"])

        first_interaction = df_temporal.groupBy("userId").agg(min("timestampHistory").alias("first_interaction"))
        df_temporal = df_temporal.join(first_interaction, on="userId", how="left") \
            .withColumn("time_since_first_interaction", current_timestamp().cast("long") - col("first_interaction"))

        self.log_step("Adding interaction score...")
        df_temporal = df_temporal.withColumn(
            "interaction_score",
            col("numberOfClicksHistory") * 0.5 +
            col("timeOnPageHistory") * 0.3 +
            col("scrollPercentageHistory") * 0.2
        )

        self.log_step("Adding recency weight...")
        df_temporal = df_temporal.withColumn(
            "recency_weight", 1 / (1 + (current_timestamp().cast("long") - col("timestampHistory")) / 86400)
        )

        self.log_step("Adding time weight...")
        avg_time_on_page = df_temporal.groupBy("history").agg(avg("timeOnPageHistory").alias("avg_time_on_page"))
        df_temporal = df_temporal.join(avg_time_on_page, on="history", how="left")
        max_time_on_page = df_temporal.agg(max("avg_time_on_page")).collect()[0][0]
        df_temporal = df_temporal.withColumn("time_weight", col("avg_time_on_page") / max_time_on_page)

        self.log_step("Calculating adjusted score...")
        df_temporal = df_temporal.withColumn(
            "adjusted_score", col("interaction_score") * col("recency_weight") * col("time_weight")
        )

        self.log_step("Writing partitioned Parquet files...")
        df_temporal.coalesce(1).write \
            .mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(output_path)

        self.log_step("'Treino' transformation completed and data saved.")

    def transform_itens(self, input_path: str, output_path: str):
        self.log_step("Starting 'Itens' transformation...")

        file_path = f"{input_path}/itens/itens/"
        self.log_step(f"Reading CSV files from {file_path}...")
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
        df = df.withColumn("issued", (to_timestamp(col("issued"), "yyyy-MM-dd HH:mm:ss").cast("long"))) \
            .withColumn("modified", (to_timestamp(col("modified"), "yyyy-MM-dd HH:mm:ss").cast("long")))

        self.log_step("Calculating days since published and modified...")
        df = df.withColumn("days_since_published",
                           ((current_timestamp().cast("long") - col("issued")) / 86400).cast("int")) \
            .withColumn("days_since_modified",
                        ((current_timestamp().cast("long") - col("modified")) / 86400).cast("int"))

        self.log_step("Dropping unused columns...")
        df = df.drop("url")

        self.log_step("Writing partitioned Parquet files...")
        df.coalesce(1).write \
            .mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(output_path)

        self.log_step("'Itens' transformation completed and data saved.")

    def merge_data(self, treino_path: str, itens_path: str, output_path: str):
        self.log_step("Starting data merge process...")

        # Ler os dados transformados
        self.log_step("Loading transformed 'Treino' dataset...")
        df_treino = self.spark.read.parquet(treino_path)

        self.log_step("Loading transformed 'Itens' dataset...")
        df_itens = self.spark.read.parquet(itens_path)

        # Realizar o join baseado na coluna "history" (do treino) e no identificador do item
        self.log_step("Merging datasets...")
        df_merged = df_treino.join(df_itens, df_treino.history == df_itens.page, how="left")

        # Selecionar apenas colunas necessárias (evitar duplicação)
        df_merged = df_merged.drop(df_itens.page)

        self.log_step("Saving merged dataset...")
        df_merged.coalesce(1).write \
            .mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(output_path)

        self.log_step("Data merge completed and saved!")


    def normalize_data(self, input_path: str, output_path: str):
        self.log_step("Starting data normalization...")

        df = self.spark.read.parquet(input_path)

        self.log_step("Applying log1p transformation...")
        log_columns = [
            "timeOnPageHistory", "interaction_score", "avg_time_on_page", "adjusted_score",
            "days_since_published", "days_since_modified", "time_since_last_interaction", "time_since_first_interaction"
        ]
        for col_name in log_columns:
            if col_name in df.columns:
                df = df.withColumn(col_name, log1p(col(col_name)))

        self.log_step("Converting 'userType' to numeric values...")
        if "userType" in df.columns:
            df = df.withColumn("userType", when(col("userType") == "Logged", 1).otherwise(0))

        self.log_step("Applying MinMaxScaler normalization...")
        minmax_columns = [
            "numberOfClicksHistory", "scrollPercentageHistory", "pageVisitsCountHistory",
            "recency_weight", "time_weight", "timestampHistory", "issued",
            "modified", "hour", "dayofweek", "month"
        ]

        normalized_columns = []  # Lista para armazenar as colunas normalizadas

        for col_name in minmax_columns:
            if col_name in df.columns:
                assembler = VectorAssembler(inputCols=[col_name], outputCol=f"{col_name}_vec")
                scaler = MinMaxScaler(inputCol=f"{col_name}_vec", outputCol=f"{col_name}_scaled")
                pipeline = Pipeline(stages=[assembler, scaler])

                self.log_step(f"Scaling feature: {col_name}")
                model = pipeline.fit(df)
                df = model.transform(df)

                normalized_columns.append(col_name)  # Adiciona à lista de colunas processadas

        self.log_step("Extracting scalar values from scaled vectors...")

        # Função UDF para extrair valores de vetores (DenseVector)
        def extract_scalar(vec):
            return float(vec[0]) if vec else None

        extract_scalar_udf = udf(extract_scalar, DoubleType())

        # Extração dos valores escalares
        for col_name in normalized_columns:
            df = df.withColumn(col_name, extract_scalar_udf(col(f"{col_name}_scaled")))

        # 🚨 **Correção: Remover as colunas APÓS a extração dos valores**
        self.log_step("Dropping vectorized columns after normalization...")
        df = df.drop(*[f"{col_name}_vec" for col_name in normalized_columns])  # Remove colunas _vec
        df = df.drop(*[f"{col_name}_scaled" for col_name in normalized_columns])  # Remove colunas _scaled

        self.log_step(f"Final columns in DataFrame after normalization: {df.columns}")

        record_count = df.count()
        self.log_step(f"Total registros após normalização: {record_count}")

        if record_count > 0:
            self.log_step("Saving normalized dataset...")
            df.coalesce(1).write \
                .mode("overwrite") \
                .option("compression", "snappy") \
                .parquet(output_path)
            self.log_step("Data normalization completed and saved!")
        else:
            self.log_step("No data available to save after normalization.")
