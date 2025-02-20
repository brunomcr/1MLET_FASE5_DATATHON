from pyspark.sql import SparkSession

class SparkSessionFactory:
    def create_spark_session(self, app_name: str):
        print("Initializing Spark Session...")
        spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.driver.host", "localhost") \
            .config("spark.driver.bindAddress", "0.0.0.0") \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "6g") \
            .config("spark.executor.cores", "2") \
            .config("spark.python.worker.memory", "2g") \
            .config("spark.default.parallelism", "4") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.memory.fraction", "0.7") \
            .config("spark.memory.storageFraction", "0.3") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
            .config("spark.sql.inMemoryColumnarStorage.batchSize", "1000") \
            .config("spark.python.worker.reuse", "true") \
            .config("spark.dynamicAllocation.enabled", "false") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000") \
            .config("spark.sql.execution.arrow.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true") \
            .config("spark.sql.files.maxPartitionBytes", "128m") \
            .config("spark.sql.shuffle.partitions", "4") \
            .master("local[*]") \
            .getOrCreate()
        
        # Desabilitar logs de warning
        spark.sparkContext.setLogLevel("ERROR")
        
        print("Spark Session initialized.")
        return spark

# .config("spark.driver.memory", "8g") \  # Increased driver memory
# .config("spark.executor.memory", "8g") \  # Increased executor memory
# .config("spark.executor.cores", "4") \  # 4 cores per executor
# .config("spark.default.parallelism", "12") \  # Optimal parallelism for reading/writing
# .config("spark.sql.shuffle.partitions", "12") \  # Reduce shuffle overhead
# .config("spark.memory.fraction", "0.8") \  # Allocate more memory to operations
# .config("spark.sql.files.maxPartitionBytes", "32m") \  # Reduce partition size
# .config("spark.local.dir", "/spark-temp") \  # Use optimized local directory for shuffle