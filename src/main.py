from utils.logger import logger
from services.downloader import Downloader
from services.file_handler import FileHandler
from services.spark_session import SparkSessionFactory
from services.transformers import BronzeToSilverTransformer
from configs.config import Config
import os
import gc
import time


def check_data_exists(config):
    """Verifica se os dados já foram processados"""
    paths = [
        f"{config.silver_path_treino}/_SUCCESS",
        f"{config.silver_path_itens}/_SUCCESS",
        f"{config.silver_path_merge}/_SUCCESS",
        f"{config.silver_path_normalizer}/_SUCCESS"
    ]

    if all(os.path.exists(path) for path in paths):
        logger.info("Todos os dados já processados. Pulando processamento.")
        return True
    return False


def main():
    logger.info("Starting ETL process...")
    
    # General configurations
    config = Config()

    # Verificar se os dados já existem
    if check_data_exists(config):
        logger.info("Dados já processados. Pulando processamento.")
        return

    # Create necessary directories
    os.makedirs(config.bronze_path, exist_ok=True)
    logger.info(f"Created directory: {config.bronze_path}")

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
    logger.info("Initializing Spark session...")
    spark_session = SparkSessionFactory().create_spark_session("ETL Process")

    try:
        # ETL process (Bronze -> Silver)
        transformer = BronzeToSilverTransformer(spark_session)
        logger.info("Starting Treino transformation...")
        transformer.transform_treino(config.bronze_path, config.silver_path_treino)
        logger.info("Starting Itens transformation...")
        transformer.transform_itens(config.bronze_path, config.silver_path_itens)
        logger.info("Starting Data Merge...")
        transformer.merge_data(config.silver_path_treino, config.silver_path_itens, config.silver_path_merge)
        logger.info("Starting Data Normalization...")
        transformer.normalize_data(config.silver_path_merge, config.silver_path_normalizer)
    finally:
        logger.info("Stopping Spark session...")
        spark_session.stop()
        time.sleep(10)
        gc.collect()
        logger.info("Spark session closed and memory released.")


if __name__ == "__main__":
    main()
