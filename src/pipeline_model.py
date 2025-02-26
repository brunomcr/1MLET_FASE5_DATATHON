from utils.logger import logger
from configs.config import Config
from services.spark_session import SparkSessionFactory
from services.train_lightfm import LightFMTrainer
import time
import gc
import os


def main():
    """Main entry point for the LightFM training pipeline.

    This function initializes the Spark session, creates an instance of the LightFMTrainer,
    loads the prepared data, splits the data into training and testing sets, trains the LightFM model,
    and saves the trained model to the specified output path. It also handles exceptions and ensures
    that the Spark session is stopped properly at the end of the process.

    Returns:
        None
    """
    logger.info("Starting model training pipeline...")

    config = Config()
    spark_session = SparkSessionFactory().create_spark_session("LightFM Training")

    try:
        trainer = LightFMTrainer(
            spark=spark_session,
            interactions_path=os.path.join(config.gold_path_lightfm_interactions, "interaction_matrix.npz"),
            user_features_path=os.path.join(config.gold_path_lightfm_user_features, "user_features.parquet"),
            item_features_path=os.path.join(config.gold_path_lightfm_item_features, "item_features.parquet"),
            model_output_path=os.path.join(config.models_path, "lightfm_model.pkl"),
            test_ratio=0.2
        )

        logger.info("Loading prepared data...")
        trainer.load_data()

        logger.info("Splitting data into train/test sets...")
        trainer.split_data()

        logger.info("Training LightFM model...")
        trainer.train_model()

        # logger.info("Evaluating model performance...")
        # metrics = trainer.evaluate_model()
        # logger.info(f"Model metrics: {metrics}")

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