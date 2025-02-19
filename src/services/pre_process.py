from utils.logger import logger
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, split, arrays_zip, from_unixtime, to_timestamp, regexp_replace,
    hour, dayofweek, month, current_timestamp, lag, avg, min, max, log1p, udf, trim, concat_ws, year, month , dayofmonth, monotonically_increasing_id
)
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, ArrayType, FloatType
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
import fasttext  # Biblioteca fastText para Python
from services.spark_session import SparkSessionFactory
import gc

class BronzeToSilverTransformer:
    def __init__(self, spark=None):
        if spark is None:
            # Usar a factory existente ao invés de criar uma nova configuração
            spark_factory = SparkSessionFactory()
            self.spark = spark_factory.create_spark_session("G1 Recommendations")
        else:
            self.spark = spark
        # UDF para extrair o valor escalar de um vetor, usada na normalização
        self.extract_scalar_udf = udf(lambda vec: float(vec[0]) if vec else None, DoubleType())

    def log_step(self, message):
        logger.info(message)

    def clean_text_columns(self, df):
        text_columns = ["title", "body", "caption"]
        for column in text_columns:
            # Remove URLs
            df = df.withColumn(column, regexp_replace(col(column), r'(https?:\/\/\S+)|(www\.\S+)', ''))
            # Remove tags HTML
            df = df.withColumn(column, regexp_replace(col(column), r'<[^>]+>', ''))
            # Remove caracteres especiais indesejados, mantendo letras, números e pontuações comuns
            df = df.withColumn(
                column,
                regexp_replace(
                    col(column),
                    r'[^A-Za-z0-9ÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÇáéíóúàèìòùâêîôûãõç\s!"#$%&\'()*+,\-./:;<=>?@\[\]\\^_`{|}~]',
                    ''
                )
            )
            # Substitui múltiplos espaços por um único espaço
            df = df.withColumn(column, regexp_replace(col(column), r'\s+', ' '))
            # Remove espaços em branco no início e final
            df = df.withColumn(column, trim(col(column)))
        return df

    def apply_fasttext_embedding(self, df, ft_model_path):
        """
        Aplica embeddings do FastText com salvamento incremental
        """
        self.log_step("Setting up fastText UDF with lazy loading...")

        def get_fasttext_embedding(text, max_retries=3):
            if text is None or not text:
                return [0.0] * 100
            
            text = text[:300]
            
            for attempt in range(max_retries):
                try:
                    global _ft_model_local
                    if '_ft_model_local' not in globals():
                        _ft_model_local = fasttext.load_model(ft_model_path)
                    return _ft_model_local.get_sentence_vector(text).tolist()
                except Exception as e:
                    if attempt == max_retries - 1:
                        return [0.0] * 100
                    time.sleep(0.1)
        
        embedding_udf = udf(get_fasttext_embedding, ArrayType(FloatType()))
        
        self.log_step("Processing embeddings in batches...")
        
        window_size = 200
        num_partitions = 4
        
        # Reparticionamento inicial
        df = df.repartition(num_partitions)
        df = df.withColumn("row_number", monotonically_increasing_id())
        df = df.withColumn("batch_id", (col("row_number") / window_size).cast("int"))
        
        batch_count = df.select("batch_id").distinct().count()
        self.log_step(f"Total number of batches: {batch_count}")
        
        # Criar diretório temporário para resultados intermediários
        temp_output_path = "temp_embeddings"
        
        for batch in range(int(batch_count)):
            try:
                self.log_step(f"Processing batch {batch + 1} of {batch_count}")
                
                batch_df = df.filter(col("batch_id") == batch)
                
                current_batch = batch_df.withColumn(
                    "fasttext_embedding",
                    embedding_udf(col("title"))
                )
                
                # Remover colunas temporárias antes de salvar
                current_batch = current_batch.drop("row_number", "batch_id")
                
                # Salvar batch atual
                current_batch.write.mode("append").parquet(f"{temp_output_path}/batch_{batch}")
                
                # Limpar cache a cada 5 batches
                if batch % 5 == 0:
                    self.spark.catalog.clearCache()
                    gc.collect()  # Forçar garbage collection
                
            except Exception as e:
                self.log_step(f"Error processing batch {batch + 1}: {str(e)}")
                continue
        
        # Ler todos os resultados salvos
        self.log_step("Combining all processed batches...")
        result_df = self.spark.read.parquet(f"{temp_output_path}/batch_*")
        
        # Otimizar particionamento final
        result_df = result_df.coalesce(num_partitions)
        
        # Limpar diretório temporário
        self.spark.sparkContext.hadoop.fs.delete(temp_output_path, True)
        
        return result_df

    def transform_treino(self, input_path: str, output_path: str):
        self.log_step("Starting 'Treino' transformation...")
        file_path = f"{input_path}/files/treino/"
        self.log_step(f"Reading CSV files from {file_path}...")
        
        df = self.spark.read.option("header", "true") \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .option("multiLine", "true") \
            .option("inferSchema", "true") \
            .csv(file_path).repartition(4)
        
        self.log_step("Finished reading CSV files.")
        
        self.log_step("Splitting and transforming columns...")
        cols_to_split = [
            "history", "timestampHistory", "numberOfClicksHistory",
            "timeOnPageHistory", "scrollPercentageHistory",
            "pageVisitsCountHistory"
        ]
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
        df_temporal = df_normalized.withColumn("hour", hour(from_unixtime(col("timestampHistory")))) \
            .withColumn("dayofweek", dayofweek(from_unixtime(col("timestampHistory")))) \
            .withColumn("month", month(from_unixtime(col("timestampHistory"))))
        window_spec = Window.partitionBy("userId").orderBy("timestampHistory")
        df_temporal = df_temporal.withColumn("time_since_last_interaction",
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
        self.log_step("Adding partition columns...")
        df = df_normalized \
            .withColumn("timestamp", from_unixtime(col("timestampHistory"))) \
            .withColumn("year", year(col("timestamp"))) \
            .withColumn("month", month(col("timestamp"))) \
            .withColumn("day", dayofmonth(col("timestamp"))) \
            .drop("timestamp")  # Remove a coluna temporária
        
        self.log_step("Writing partitioned Parquet files for treino...")
        df.write.mode("overwrite") \
               .option("compression", "snappy") \
               .option("maxRecordsPerFile", "10000") \
               .partitionBy("year", "month", "day") \
               .parquet(output_path)
        self.log_step("'Treino' transformation completed and data saved.")

    def transform_itens(self, input_path: str, output_path: str, model_path: str):
        self.log_step("Starting 'Itens' transformation...")
        file_path = f"{input_path}/itens/itens/"
        self.log_step(f"Reading CSV files from {file_path}...")
        
        # Reduzir o número de partições
        df = self.spark.read.option("header", "true") \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .option("multiLine", "true") \
            .option("inferSchema", "true") \
            .csv(file_path).repartition(4)  # Reduzido de 20 para 4
        
        self.log_step("Finished reading CSV files.")

        self.log_step("Cleaning up timestamp columns...")
        df = df.withColumn("issued", regexp_replace(col("issued"), r"\+00:00", "")) \
               .withColumn("modified", regexp_replace(col("modified"), r"\+00:00", ""))
        
        self.log_step("Converting strings to timestamp columns...")
        df = df.withColumn("issued", to_timestamp(col("issued"), "yyyy-MM-dd HH:mm:ss")) \
               .withColumn("modified", to_timestamp(col("modified"), "yyyy-MM-dd HH:mm:ss"))
        
        self.log_step("Calculating days since published and modified...")
        df = df.withColumn("days_since_published",
                           ((current_timestamp().cast("long") - col("issued").cast("long")) / 86400).cast("int")) \
               .withColumn("days_since_modified",
                           ((current_timestamp().cast("long") - col("modified").cast("long")) / 86400).cast("int"))
        
        self.log_step("Cleaning text columns (title, body, caption)...")
        df = self.clean_text_columns(df)

        self.log_step("Dropping unused columns...")
        df = df.drop("url")
        df = df.drop("body")
        df = df.drop("caption")

        # self.log_step("Applying fastText embeddings on items...")
        # df = self.apply_fasttext_embedding(df, model_path)
        
        if df is None:
            self.log_step("Error: FastText embedding failed")
            return
        
        self.log_step("Adding partition columns...")
        df = df.withColumn("year", year(col("issued"))) \
               .withColumn("month", month(col("issued"))) \
               .withColumn("day", dayofmonth(col("issued")))

        self.log_step("Writing partitioned Parquet files for itens...")
        df.write.mode("overwrite") \
               .option("compression", "snappy") \
               .option("maxRecordsPerFile", "10000") \
               .partitionBy("year", "month", "day") \
               .parquet(output_path)
        
        self.log_step("'Itens' transformation completed and data saved.")

    def normalize_treino(self, input_path: str, output_path: str):
        self.log_step("Starting treino normalization...")
        df = self.spark.read.parquet(input_path)
        self.log_step("Applying log1p transformation...")
        log_columns = ["timeOnPageHistory", "time_since_last_interaction", "time_since_first_interaction"]
        for col_name in log_columns:
            if col_name in df.columns:
                df = df.withColumn(col_name, log1p(col(col_name)))
        self.log_step("Applying MinMaxScaler for treino...")
        minmax_columns = ["numberOfClicksHistory", "scrollPercentageHistory", "pageVisitsCountHistory", "hour", "dayofweek", "month"]
        for col_name in minmax_columns:
            if col_name in df.columns:
                assembler = VectorAssembler(inputCols=[col_name], outputCol=f"{col_name}_vec")
                scaler = MinMaxScaler(inputCol=f"{col_name}_vec", outputCol=f"{col_name}_scaled")
                pipeline = Pipeline(stages=[assembler, scaler])
                model = pipeline.fit(df)
                df = model.transform(df)
                df = df.withColumn(col_name, self.extract_scalar_udf(col(f"{col_name}_scaled")))
                df = df.drop(f"{col_name}_vec").drop(f"{col_name}_scaled")
        
        self.log_step("Saving normalized treino dataset...")
        df.write.mode("overwrite").option("compression", "snappy").parquet(output_path)
        self.log_step("Treino normalization completed!")

    def normalize_itens(self, input_path: str, output_path: str):
        self.log_step("Starting itens normalization...")
        df = self.spark.read.parquet(input_path)
        self.log_step("Applying log1p transformation for itens...")
        log_columns = ["days_since_published", "days_since_modified"]
        for col_name in log_columns:
            if col_name in df.columns:
                df = df.withColumn(col_name, log1p(col(col_name)))
        self.log_step("Applying MinMaxScaler for itens...")
        minmax_columns = ["days_since_published", "days_since_modified"]
        for col_name in minmax_columns:
            if col_name in df.columns:
                assembler = VectorAssembler(inputCols=[col_name], outputCol=f"{col_name}_vec")
                scaler = MinMaxScaler(inputCol=f"{col_name}_vec", outputCol=f"{col_name}_scaled")
                pipeline = Pipeline(stages=[assembler, scaler])
                model = pipeline.fit(df)
                df = model.transform(df)
                df = df.withColumn(col_name, self.extract_scalar_udf(col(f"{col_name}_scaled")))
                df = df.drop(f"{col_name}_vec").drop(f"{col_name}_scaled")
        self.log_step("Saving normalized itens dataset...")
        df.write.mode("overwrite").option("compression", "snappy").parquet(output_path)
        self.log_step("Itens normalization completed!")
