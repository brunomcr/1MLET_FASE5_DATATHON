from pyspark.sql.functions import col, udf, year, month, dayofmonth
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
from pyspark.ml import Pipeline
import time
from utils.logger import logger
import os

class TextProcessor:
    def __init__(self, spark, vector_size=100):
        self.spark = spark
        self.vector_size = vector_size
        self.model = None
        
    def log_step(self, message):
        logger.info(message)
        
    def create_pipeline(self):
        """Criar pipeline de processamento de texto"""
        tokenizer = Tokenizer(inputCol="title", outputCol="words")
        
        # Remover stopwords em português
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", 
                                 stopWords=self._get_portuguese_stopwords())
        
        # Usar HashingTF para feature hashing (mais eficiente que CountVectorizer)
        hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features",
                             numFeatures=self.vector_size)
        
        idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
        
        return Pipeline(stages=[tokenizer, remover, hashingTF, idf])
    
    def _get_portuguese_stopwords(self):
        """Lista de stopwords em português"""
        # Adicionar stopwords comuns em português
        return ["a", "o", "e", "é", "de", "do", "da", "em", "para", "com", "um", "uma", 
                "os", "as", "que", "no", "na", "por", "mais", "das", "dos", "ao", "ou",
                "são", "dos", "como", "mas", "foi", "ao", "ele", "dela", "esse", "essa",
                "pelo", "pela", "até", "isso", "ela", "entre", "depois", "sem", "mesmo",
                "aos", "seus", "quem", "nas", "me", "esse", "essa", "esses", "essas",
                "seu", "sua", "seus", "suas", "só"]
    
    def process_itens(self, input_path: str, output_path: str):
        """
        Processa os itens da camada silver aplicando TF-IDF
        """
        self.log_step("Starting TF-IDF processing for items...")
        start_time = time.time()
        
        try:
            # Criar diretório de saída se não existir
            os.makedirs(output_path, exist_ok=True)
            
            # Ler dados da camada silver
            self.log_step("Reading input data...")
            df = self.spark.read.parquet(input_path)
            
            # Criar e treinar pipeline
            self.log_step("Training TF-IDF model...")
            pipeline = self.create_pipeline()
            self.model = pipeline.fit(df)
            
            # Processar em anos
            years = df.select("year").distinct().collect()
            for year_row in years:
                year_val = year_row['year']
                self.log_step(f"Processing year {year_val}...")
                
                year_df = df.filter(col("year") == year_val)
                processed_year = self.model.transform(year_df)
                
                # Salvar por ano
                self.log_step(f"Saving data for year {year_val}...")
                (processed_year
                 .write
                 .mode("append")
                 .option("compression", "snappy")
                 .partitionBy("year", "month", "day")
                 .parquet(output_path))
                
                # Limpar cache
                self.spark.catalog.clearCache()
            
            elapsed_time = time.time() - start_time
            self.log_step(f"Text processing completed in {elapsed_time:.2f} seconds!")
            
        except Exception as e:
            self.log_step(f"Error in text processing: {str(e)}")
            raise

    def validate_results(self, output_path: str):
        """
        Validar resultados do processamento TF-IDF
        """
        self.log_step("Validando resultados...")
        
        # Ler dados processados
        df = self.spark.read.parquet(output_path)
        
        # Mostrar schema
        self.log_step("Schema dos dados processados:")
        df.printSchema()
        
        # Contar registros
        total_records = df.count()
        self.log_step(f"Total de registros processados: {total_records}")
        
        # Mostrar exemplo de um registro
        self.log_step("Exemplo de registro processado:")
        df.select("title", "tfidf_features").show(1, truncate=False)
        
        # Estatísticas básicas
        self.log_step("Distribuição por ano:")
        df.groupBy("year").count().show() 