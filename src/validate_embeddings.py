from services.spark_session import SparkSessionFactory
from utils.logger import logger

def validate_embeddings():
    spark = None
    try:
        logger.info("Starting validation...")
        
        # Criar sessão Spark
        spark_factory = SparkSessionFactory()
        spark = spark_factory.create_spark_session("Validation")
        
        # Ler dados
        df = spark.read.parquet("/app/datalake/silver/itens_embeddings")
        
        # Mostrar apenas informações essenciais
        logger.info(f"Total records: {df.count()}")
        logger.info("\nDistribution by year:")
        df.groupBy("year").count().orderBy("year").show()
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    validate_embeddings() 