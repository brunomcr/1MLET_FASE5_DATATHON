from pyspark.sql import SparkSession

class SparkSessionFactory:
    def create_spark_session(self, app_name: str):
        print("Initializing Spark Session...")
        spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.driver.memory", "10g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.executor.cores", "6") \
            .config("spark.default.parallelism", "12") \
            .config("spark.sql.shuffle.partitions", "50") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.ui.enabled", "false") \
            .config("spark.sql.files.maxPartitionBytes", "32m") \
            .getOrCreate()
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