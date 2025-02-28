from utils.logger import logger
from configs.config import Config
from services.spark_session import SparkSessionFactory
from services.train_lightfm import LightFMTrainer
import time
import gc
import os
import argparse
import numpy as np


def main():
    # Configurar o parser de argumentos
    parser = argparse.ArgumentParser(description='Train LightFM recommendation model')
    parser.add_argument('--sample_size', type=float, default=100.0,
                        help='Percentage of data to use for training (e.g., 1.0 for 1%, 10.0 for 10%, 100.0 for full dataset)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    
    # Parse os argumentos
    args = parser.parse_args()
    
    logger.info("Starting model training pipeline...")
    logger.info(f"Using {args.sample_size}% of the dataset")
    logger.info(f"Training for {args.epochs} epochs")

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

        # Carrega os dados preparados
        logger.info("Loading prepared data...")
        trainer.load_data()
        
        # Aplicar amostragem se sample_size < 100%
        if args.sample_size < 100.0:
            logger.info(f"Sampling {args.sample_size}% of the data...")
            # Obter uma amostra do conjunto de dados
            trainer.sample_data(args.sample_size / 100.0)

        # Divide em treino e teste
        logger.info("Splitting data into train/test sets...")
        trainer.split_data()

        # Treina o modelo com o número de épocas especificado
        logger.info(f"Training LightFM model with {args.epochs} epochs...")
        trainer.train_model(epochs=args.epochs)

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