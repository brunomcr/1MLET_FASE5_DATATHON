from utils.logger import logger
from configs.config import Config
from services.spark_session import SparkSessionFactory
from services.train_lightfm import LightFMTrainer
import time
import gc


def main():
    logger.info("Starting model training pipeline...")

    config = Config()
    spark_session = SparkSessionFactory().create_spark_session("LightFM Training")

    try:
        trainer = LightFMTrainer(
            spark=spark_session,
            interactions_path=f"{config.gold_path_lightfm_interactions}/interaction_matrix.npz",
            user_features_path=f"{config.gold_path_lightfm_user_features}/user_features.parquet",
            item_features_path=f"{config.gold_path_lightfm_item_features}/item_features.parquet",
            model_output_path=f"{config.models_path}/lightfm_model.pkl"
        )

        # Carrega os dados preparados
        logger.info("Loading prepared data...")
        trainer.load_data()

        # Divide em treino e teste
        logger.info("Splitting data into train/test sets...")
        trainer.split_data()

        # Treina o modelo
        logger.info("Training LightFM model...")
        trainer.train_model()

        # Avalia o modelo
        logger.info("Evaluating model performance...")
        metrics = trainer.evaluate_model()
        logger.info(f"Model metrics: {metrics}")

        # Salva o modelo treinado
        logger.info("Saving trained model...")
        trainer.save_model()

        logger.info("LightFM training pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Error in LightFM Training: {str(e)}")
        raise
    finally:
        logger.info("Stopping Spark session...")
        spark_session.stop()
        time.sleep(5)
        gc.collect()


if __name__ == "__main__":
    main() 