from utils.logger import logger
from services.downloader import Downloader
from services.file_handler import FileHandler
from services.spark_session import SparkSessionFactory
from services.pre_process import BronzeToSilverTransformer
from services.text_processor import TextProcessor
from services.feature_engineering import FeatureEngineering
from services.advanced_features import AdvancedFeatureEngineering
from configs.config import Config
import os
import gc
import time
import shutil


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
    logger.info(f"Created directory: {config.bronze_path}")

    # # Baixar o arquivo
    # print("**************************************************")
    # print("ETAPA 0: Download Files **************************")
    # print("**************************************************")
    # downloader = Downloader()
    # downloader.download_file(
    #     config.download_url,
    #     config.output_file
    # )

    # # Descompactar e deletar o .zip
    # print("**************************************************")
    # print("ETAPA 0: Unzip Files *****************************")
    # print("**************************************************")
    # file_handler = FileHandler()
    # file_handler.unzip_and_delete(config.output_file, config.bronze_path)

    # # Etapa 1: Bronze to Silver + Normalização
    # print("**************************************************")
    # print("ETAPA 1: Bronze to Silver + Normalização *********")
    # print("**************************************************")
    # spark_session = SparkSessionFactory().create_spark_session("Bronze to Silver ETL")
    # try:
    #     transformer = BronzeToSilverTransformer(spark_session)
        
    #     # 1.1 Transformação básica
    #     transformer.transform_treino(config.bronze_path, config.silver_path_treino)
    #     transformer.transform_itens(config.bronze_path, config.silver_path_itens)
        
    #     # 1.2 Normalização
    #     transformer.normalize_treino(config.silver_path_treino, config.silver_path_treino_normalized)
    #     transformer.normalize_itens(config.silver_path_itens, config.silver_path_itens_normalized)
    # finally:
    #     spark_session.stop()
    #     gc.collect()


    # # Etapa 2: Processamento de Texto
    # print("**************************************************")
    # print("ETAPA 2: Processamento de Texto ******************")
    # print("**************************************************")
    # spark_session = SparkSessionFactory().create_spark_session("Text Processing")
    # try:
    #     text_processor = TextProcessor(spark_session)
    #     text_processor.process_itens(
    #         input_path=config.silver_path_itens_normalized,
    #         output_path=config.silver_path_itens_embeddings
    #     )
    # finally:
    #     spark_session.stop()
    #     gc.collect()

    # Etapa 3: Feature Engineering Avançado
    print("**************************************************")
    print("ETAPA 3: Feature Engineering Avançado ************")
    print("**************************************************")
    spark_session = SparkSessionFactory().create_spark_session("Advanced Features")
    try:
        advanced_features = AdvancedFeatureEngineering(spark_session)
        features_df = advanced_features.process_features(
            treino_df=spark_session.read.parquet(config.silver_path_treino_normalized),
            items_df=spark_session.read.parquet(config.silver_path_itens_embeddings)
        )
        
        features_df.write.mode("overwrite") \
            .partitionBy("year", "month", "day") \
            .parquet(config.gold_path_advanced_features)
    finally:
        spark_session.stop()
        gc.collect()

    # Etapa 4: Preparação Final para LightFM
    print("**************************************************")
    print("ETAPA 4: Preparação Final para LightFM ***********")
    print("**************************************************")
    spark_session = SparkSessionFactory().create_spark_session("LightFM Prep")
    try:
        feature_engineering = FeatureEngineering(spark_session)
        stats = feature_engineering.prepare_lightfm_matrices(
            treino_path=config.silver_path_treino_normalized,
            items_path=config.silver_path_itens_embeddings,
            advanced_features_path=config.gold_path_advanced_features,
            output_path=config.gold_path_matrices
        )
        logger.info(f"Feature Engineering completed with stats: {stats}")
    finally:
        spark_session.stop()
        gc.collect()


if __name__ == "__main__":
    main()
