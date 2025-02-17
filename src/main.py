from utils.logger import logger
from services.downloader import Downloader
from services.file_handler import FileHandler
from services.spark_session import SparkSessionFactory
from services.pre_process import BronzeToSilverTransformer
from configs.config import Config
import os
import gc
import time


def check_data_exists(config):
    """Verifica se os dados já foram processados"""
    paths = [
        f"{config.silver_path_treino}/_SUCCESS",
        f"{config.silver_path_itens}/_SUCCESS",
        f"{config.silver_path_treino_normalized}/_SUCCESS",
        f"{config.silver_path_itens_normalized}/_SUCCESS"
    ]

    if all(os.path.exists(path) for path in paths):
        logger.info("Todos os dados já foram processados. Pulando processamento.")
        return True
    return False


def main():
    logger.info("Starting ETL process...")

    # Configurações gerais
    config = Config()

    # Verificar se os dados já existem
    if check_data_exists(config):
        logger.info("Dados já processados. Pulando processamento.")
        return

    # Criar diretórios necessários
    os.makedirs(config.bronze_path, exist_ok=True)
    logger.info(f"Created directory: {config.bronze_path}")

    # Baixar o arquivo
    downloader = Downloader()
    downloader.download_file(
        config.download_url,
        config.output_file
    )

    # Descompactar e deletar o .zip
    file_handler = FileHandler()
    file_handler.unzip_and_delete(config.output_file, config.bronze_path)

    # Iniciar a sessão Spark
    logger.info("Initializing Spark session...")
    spark_session = SparkSessionFactory().create_spark_session("ETL Process")

    try:
        transformer = BronzeToSilverTransformer(spark_session)

        # --------------------------------------
        # TRANSFORMAR OS DADOS
        # --------------------------------------
        logger.info("Starting Treino transformation...")
        transformer.transform_treino(config.bronze_path, config.silver_path_treino)

        logger.info("Starting Itens transformation...")
        transformer.transform_itens(config.bronze_path, config.silver_path_itens)

        # --------------------------------------
        # NORMALIZAR OS DADOS
        # --------------------------------------
        logger.info("Starting Treino normalization...")
        transformer.normalize_treino(config.silver_path_treino, config.silver_path_treino_normalized)

        logger.info("Starting Itens normalization...")
        transformer.normalize_itens(config.silver_path_itens, config.silver_path_itens_normalized)

    finally:
        logger.info("Stopping Spark session...")
        spark_session.stop()
        time.sleep(10)
        gc.collect()
        logger.info("Spark session closed and memory released.")


if __name__ == "__main__":
    main()
