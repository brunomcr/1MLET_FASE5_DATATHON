from services.downloader import Downloader
from services.file_handler import FileHandler
from services.spark_session import SparkSessionFactory
from services.transformers import BronzeToSilverTransformer
from configs.config import Config
import os

def main():
    # General configurations
    config = Config()

    # Create necessary directories
    os.makedirs(config.bronze_path, exist_ok=True)

    # Download the file
    downloader = Downloader()
    downloader.download_file(
        config.download_url,
        config.output_file
    )

    # Unzip and delete the .zip file
    file_handler = FileHandler()
    file_handler.unzip_and_delete(config.output_file, config.bronze_path)

    # Start Spark session
    spark_session = SparkSessionFactory().create_spark_session("ETL Process")

    try:
        # ETL process (Bronze -> Silver)
        transformer = BronzeToSilverTransformer(spark_session)
        transformer.transform_treino(config.bronze_path, config.silver_path_treino)
        transformer.transform_itens(config.bronze_path, config.silver_path_itens)
    finally:
        # Ensure Spark session is stopped
        print("Stopping Spark session...")
        spark_session.stop()
        print("Spark session stopped.")

if __name__ == "__main__":
    main()
