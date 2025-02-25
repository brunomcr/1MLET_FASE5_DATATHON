import logging
from pyspark.sql import SparkSession
from app.core.config import Settings


logger = logging.getLogger(__name__)



class SparkSessionFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    def create_spark_session(self, app_name: str) -> SparkSession:
        spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.driver.memory", self.settings.SPARK_DRIVER_MEMORY) \
            .config("spark.executor.memory", self.settings.SPARK_EXECUTOR_MEMORY) \
            .config("spark.sql.shuffle.partitions", self.settings.SPARK_SQL_SHUFFLE_PARTITIONS) \
            .config("spark.memory.fraction", self.settings.SPARK_MEMORY_FRACTION) \
            .config("spark.sql.files.maxPartitionBytes", self.settings.SPARK_SQL_FILES_MAX_PARTITION_BYTES) \
            .config("spark.sql.adaptive.enabled", self.settings.SPARK_SQL_ADAPTIVE_ENABLED) \
            .config("spark.sql.adaptive.coalescePartitions.enabled", self.settings.SPARK_SQL_ADAPTIVE_COALESCE_PARTITIONS_ENABLED) \
            .config("spark.sql.adaptive.skewJoin.enabled", self.settings.SPARK_SQL_ADAPTIVE_SKEW_JOIN_ENABLED) \
            .config("spark.driver.host", self.settings.SPARK_DRIVER_HOST) \
            .config("spark.driver.bindAddress", self.settings.SPARK_DRIVER_BIND_ADDRESS) \
            .config("spark.network.timeout", self.settings.SPARK_NETWORK_TIMEOUT) \
            .config("spark.cleaner.periodicGC.interval", self.settings.SPARK_CLEANER_PERIODIC_GC_INTERVAL) \
            .config("spark.sql.files.openCostInBytes", self.settings.SPARK_SQL_FILES_OPEN_COST_IN_BYTES) \
            .config("spark.sql.broadcastTimeout", self.settings.SPARK_SQL_BROADCAST_TIMEOUT) \
            .config("spark.sql.parquet.filterPushdown", self.settings.SPARK_SQL_PARQUET_FILTER_PUSHDOWN) \
            .config("spark.sql.inMemoryColumnarStorage.compressed", self.settings.SPARK_SQL_IN_MEMORY_COLUMNAR_STORAGE_COMPRESSED) \
            .config("spark.sql.parquet.writeLegacyFormat", self.settings.SPARK_SQL_PARQUET_WRITE_LEGACY_FORMAT) \
            .config("spark.default.parallelism", self.settings.SPARK_DEFAULT_PARALLELISM) \
            .master("local[*]") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel(self.settings.SPARK_LOG_LEVEL)
        logger.info("Spark Session initialized.")
        
        return spark