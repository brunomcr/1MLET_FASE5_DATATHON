import pytest
from unittest.mock import patch, Mock, MagicMock
import logging

from app.ml.utils.spark_session_factory import SparkSessionFactory
from app.core.config import Settings


class TestSparkSessionFactory:
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with all required Spark configurations"""
        settings = Mock(spec=Settings)
        
        # Configure all required settings
        settings.SPARK_DRIVER_MEMORY = "4g"
        settings.SPARK_EXECUTOR_MEMORY = "2g"
        settings.SPARK_SQL_SHUFFLE_PARTITIONS = "200"
        settings.SPARK_MEMORY_FRACTION = "0.8"
        settings.SPARK_SQL_FILES_MAX_PARTITION_BYTES = "134217728"
        settings.SPARK_SQL_ADAPTIVE_ENABLED = True
        settings.SPARK_SQL_ADAPTIVE_COALESCE_PARTITIONS_ENABLED = True
        settings.SPARK_SQL_ADAPTIVE_SKEW_JOIN_ENABLED = True
        settings.SPARK_DRIVER_HOST = "localhost"
        settings.SPARK_DRIVER_BIND_ADDRESS = "localhost"
        settings.SPARK_NETWORK_TIMEOUT = "120s"
        settings.SPARK_CLEANER_PERIODIC_GC_INTERVAL = "30min"
        settings.SPARK_SQL_FILES_OPEN_COST_IN_BYTES = "4194304"
        settings.SPARK_SQL_BROADCAST_TIMEOUT = "300s"
        settings.SPARK_SQL_PARQUET_FILTER_PUSHDOWN = True
        settings.SPARK_SQL_IN_MEMORY_COLUMNAR_STORAGE_COMPRESSED = True
        settings.SPARK_SQL_PARQUET_WRITE_LEGACY_FORMAT = False
        settings.SPARK_DEFAULT_PARALLELISM = "8"
        settings.SPARK_LOG_LEVEL = "WARN"
        
        return settings
    
    @pytest.fixture
    def factory(self, mock_settings):
        """Create a SparkSessionFactory with mock settings"""
        return SparkSessionFactory(settings=mock_settings)
    
    @patch('app.ml.utils.spark_session_factory.SparkSession')
    def test_create_spark_session(self, mock_spark_session_class, factory, mock_settings, caplog):
        """Test creating a Spark session with all configurations"""
        # Setup mock builder chain
        mock_spark = MagicMock()
        mock_builder = MagicMock()
        
        # Setup the builder chain to return itself for method chaining
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.master.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark
        
        # Set up the SparkSession.builder to return our mock builder
        mock_spark_session_class.builder = mock_builder
        
        # Set up logging capture
        caplog.set_level(logging.INFO)
        
        # Call the method under test
        result = factory.create_spark_session("test_app")
        
        # Verify the result is the mock spark session
        assert result == mock_spark
        
        # Verify builder was called with app name
        mock_builder.appName.assert_called_once_with("test_app")
        
        # Verify master was set to local[*]
        mock_builder.master.assert_called_once_with("local[*]")
        
        # Verify getOrCreate was called
        mock_builder.getOrCreate.assert_called_once()
        
        # Verify log level was set
        mock_spark.sparkContext.setLogLevel.assert_called_once_with(mock_settings.SPARK_LOG_LEVEL)
        
        # Verify initialization was logged
        assert "Spark Session initialized" in caplog.text
        
        # Verify config was called for each setting (at least 18 times)
        assert mock_builder.config.call_count >= 18
    
    @patch('app.ml.utils.spark_session_factory.SparkSession')
    def test_create_spark_session_with_custom_app_name(self, mock_spark_session_class, factory):
        """Test creating a Spark session with a custom app name"""
        # Setup mock builder chain
        mock_spark = MagicMock()
        mock_builder = MagicMock()
        
        # Setup the builder chain to return itself for method chaining
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.master.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark
        
        # Set up the SparkSession.builder to return our mock builder
        mock_spark_session_class.builder = mock_builder
        
        # Call the method with a custom app name
        factory.create_spark_session("custom_app_name")
        
        # Verify the app name was set correctly
        mock_builder.appName.assert_called_once_with("custom_app_name")
    
    @patch('app.ml.utils.spark_session_factory.SparkSession')
    def test_create_spark_session_exception_handling(self, mock_spark_session_class, factory, caplog):
        """Test exception handling during Spark session creation"""
        # Setup mock builder chain
        mock_builder = MagicMock()
        
        # Setup the builder chain to return itself for method chaining
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.master.return_value = mock_builder
        
        # Make getOrCreate raise an exception
        mock_builder.getOrCreate.side_effect = Exception("Spark session creation failed")
        
        # Set up the SparkSession.builder to return our mock builder
        mock_spark_session_class.builder = mock_builder
        
        # Set up logging capture
        caplog.set_level(logging.ERROR)
        
        # Call the method and expect the exception to propagate
        with pytest.raises(Exception) as exc_info:
            factory.create_spark_session("test_app")
        
        # Verify the exception message
        assert "Spark session creation failed" in str(exc_info.value) 