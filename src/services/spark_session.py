from pyspark.sql import SparkSession

class SparkSessionFactory:
    def create_spark_session(self, app_name: str):
        print("Initializing Spark Session...")
        spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.sql.shuffle.partitions", "10") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.sql.files.maxPartitionBytes", "128m") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.driver.host", "localhost") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.network.timeout", "800s") \
            .config("spark.cleaner.periodicGC.interval", "1min") \
            .config("spark.sql.files.openCostInBytes", "1048576") \
            .config("spark.sql.broadcastTimeout", "300") \
            .config("spark.sql.parquet.filterPushdown", "true") \
            .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
            .config("spark.sql.parquet.writeLegacyFormat", "false") \
            .config("spark.default.parallelism", "10") \
            .master("local[*]") \
            .getOrCreate()
        

        spark.sparkContext.setLogLevel("WARN")
        print("Spark Session initialized.")
        return spark

