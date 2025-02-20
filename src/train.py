from services.spark_session import SparkSessionFactory
from services.model_trainer import LightFMTrainer
from configs.config import Config
from utils.logger import logger
import os

def main():
    # Inicializar Spark
    spark = SparkSessionFactory().create_spark_session("LightFM Training")
    config = Config()
    
    try:
        logger.info("=== Iniciando processo de treinamento ===")
        
        # Verificar se o diretório de matrizes existe
        matrices_path = config.gold_path_matrices
        if not os.path.exists(matrices_path):
            logger.error(f"Diretório de matrizes não encontrado: {matrices_path}")
            raise FileNotFoundError(f"Diretório não encontrado: {matrices_path}")
        
        logger.info("1. Criando trainer...")
        trainer = LightFMTrainer(spark)
        
        logger.info("2. Preparando dados...")
        interactions, item_features = trainer.prepare_matrices(
            interactions_path=config.train_interactions_path,
            item_features_path=config.gold_path_item_features
        )
        
        # Adicionar informações sobre as matrizes
        logger.info(f"Dimensões das matrizes:")
        logger.info(f"- Interações: {interactions.shape}")
        logger.info(f"- Features dos itens: {item_features.shape}")
        
        # Dividir em treino/teste (80/20)
        train_size = int(0.8 * interactions.shape[0])
        train = interactions[:train_size]
        test = interactions[train_size:]
        logger.info(f"3. Dados divididos em:")
        logger.info(f"- Treino: {train.shape}")
        logger.info(f"- Teste: {test.shape}")
        
        # Treinar modelo
        logger.info("4. Iniciando treinamento...")
        trainer.train(
            interactions=train,
            item_features=item_features,
            epochs=30
        )
        
        # Avaliar
        logger.info("5. Avaliando modelo...")
        metrics = trainer.evaluate(test)
        
        # Salvar modelo
        model_path = f"{config.gold_path_models}/lightfm_model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Garantir que o diretório existe
        logger.info(f"6. Salvando modelo em {model_path}")
        trainer.save_model(model_path)
        
        logger.info("=== Treinamento concluído com sucesso! ===")
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 