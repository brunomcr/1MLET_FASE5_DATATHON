from utils.logger import logger
from services.downloader import Downloader
from services.file_handler import FileHandler
from services.spark_session import SparkSessionFactory
from services.pre_process import BronzeToSilverTransformer
from services.text_processor import TFIDFProcessor
from services.feature_engineering import FeatureEngineering
from configs.config import Config
import os
import gc
import time
import shutil
from services.prepare_lightfm_data import LightFMDataPreparer
from pyspark.sql import SparkSession


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

    config = Config()

    # Criar diretórios necessários
    os.makedirs(config.bronze_path, exist_ok=True)
    os.makedirs(config.silver_path, exist_ok=True)
    os.makedirs(config.gold_path, exist_ok=True)
    os.makedirs(config.models_path, exist_ok=True)
    logger.info(f"Created directories: {config.bronze_path}, {config.silver_path}, {config.gold_path}, {config.models_path}")

    # Criar diretório se não existir
    os.makedirs(config.gold_path_lightfm_interactions, exist_ok=True)
    os.makedirs(config.gold_path_lightfm_user_features, exist_ok=True)
    os.makedirs(config.gold_path_lightfm_item_features, exist_ok=True)

    # # Baixar o arquivo
    # downloader = Downloader()
    # downloader.download_file(
    #     config.download_url,
    #     config.output_file
    # )
    #
    # # Descompactar e deletar o .zip
    # file_handler = FileHandler()
    # file_handler.unzip_and_delete(config.output_file, config.bronze_path)

    # # Etapa 1: Processamento Bronze to Silver
    # logger.info("Starting Bronze to Silver transformation...")
    # spark_session = SparkSessionFactory().create_spark_session("Bronze to Silver ETL")
    #
    # try:
    #     transformer = BronzeToSilverTransformer(spark_session)
    #
    #     # Transformar dados
    #     logger.info("Starting Treino transformation...")
    #     transformer.transform_treino(config.bronze_path, config.silver_path_treino)
    #
    #     logger.info("Starting Itens transformation...")
    #     transformer.transform_itens(config.bronze_path, config.silver_path_itens)
    #
    #     # Normalizar dados
    #     logger.info("Starting Treino normalization...")
    #     transformer.normalize_treino(config.silver_path_treino, config.silver_path_treino_normalized)
    #
    #     logger.info("Starting Itens normalization...")
    #     try:
    #         transformer.normalize_itens(config.silver_path_itens, config.silver_path_itens_normalized)
    #         spark_session.catalog.clearCache()
    #     except Exception as e:
    #         logger.error(f"Error saving normalized data: {str(e)}")
    #         raise
    # finally:
    #     logger.info("Stopping first Spark session...")
    #     spark_session.stop()
    #     time.sleep(5)
    #     gc.collect()
    #
    # # Etapa 2: Processamento de Texto
    # logger.info("Starting Text processing...")
    # spark_session = SparkSessionFactory().create_spark_session("Text Processing")
    #
    # try:
    #     text_processor = TFIDFProcessor(spark_session)
    #
    #     # Processar dados, passando o caminho de saída
    #     output_path = f"{config.silver_path_itens_embeddings}"
    #     text_processor.process(output_path)
    #
    # except Exception as e:
    #     logger.error(f"Error in Text processing: {str(e)}")
    #     raise
    # finally:
    #     logger.info("Stopping Spark session...")
    #     spark_session.stop()
    #     time.sleep(5)
    #     gc.collect()

    # Etapa 3: Preparação dos dados para LightFM
    logger.info("Starting LightFM data preparation...")
    spark_session = SparkSessionFactory().create_spark_session("LightFM Data Preparation")
    try:
        preparer = LightFMDataPreparer(spark_session, config.silver_path_treino_normalized, config.silver_path_itens_embeddings,
                                       config.gold_path_lightfm_interactions, config.gold_path_lightfm_user_features,
                                       config.gold_path_lightfm_item_features)
        preparer.prepare_data()
    except Exception as e:
        logger.error(f"Error in LightFM Data Preparation: {str(e)}")
        raise
    finally:
        logger.info("Stopping Spark session...")
        spark_session.stop()
        time.sleep(5)
        gc.collect()

    logger.info("ETL process completed successfully!")


if __name__ == "__main__":
    main()
