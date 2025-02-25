from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    ALLOWED_HOSTS: List[str]

    # Model Paths
    MODEL_PATH: str
    USER_FEATURES_PATH: str
    ITEM_FEATURES_PATH: str

    # Spark Configuration
    SPARK_DRIVER_MEMORY: str
    SPARK_EXECUTOR_MEMORY: str
    SPARK_SQL_SHUFFLE_PARTITIONS: str
    SPARK_MEMORY_FRACTION: str
    SPARK_SQL_FILES_MAX_PARTITION_BYTES: str
    SPARK_SQL_ADAPTIVE_ENABLED: bool
    SPARK_SQL_ADAPTIVE_COALESCE_PARTITIONS_ENABLED: bool
    SPARK_SQL_ADAPTIVE_SKEW_JOIN_ENABLED: bool
    SPARK_DRIVER_HOST: str
    SPARK_DRIVER_BIND_ADDRESS: str
    SPARK_NETWORK_TIMEOUT: str
    SPARK_CLEANER_PERIODIC_GC_INTERVAL: str
    SPARK_SQL_FILES_OPEN_COST_IN_BYTES: str
    SPARK_SQL_BROADCAST_TIMEOUT: str
    SPARK_SQL_PARQUET_FILTER_PUSHDOWN: bool
    SPARK_SQL_IN_MEMORY_COLUMNAR_STORAGE_COMPRESSED: bool
    SPARK_SQL_PARQUET_WRITE_LEGACY_FORMAT: bool
    SPARK_DEFAULT_PARALLELISM: str
    SPARK_LOG_LEVEL: str

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore'
    )


# Initialize settings
settings = Settings() 