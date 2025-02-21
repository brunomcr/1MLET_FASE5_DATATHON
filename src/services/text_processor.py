from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, explode, split, year, month, dayofmonth, trim
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline
from utils.logger import logger
from configs.config import Config


class TFIDFProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = Config()

        # Lista de stop words em português
        self.portuguese_stop_words = [
            "a", "o", "as", "os", "um", "uma", "para", "com", "de", "do", "da", "em", "no", "na",
            "e", "que", "é", "do", "dos", "das", "se", "por", "como", "mas", "mais", "ou", "se",
            "não", "já", "tudo", "todos", "todas", "nada", "algum", "alguma", "alguns", "algumas",
            "quem", "onde", "quando", "como", "porque", "pelo", "pela", "pelo", "pelo", "pelo",
            "meu", "minha", "teu", "tua", "seu", "sua", "nosso", "nossa", "deles", "delas", "isso",
            "aquilo", "isto", "aqui", "ali", "lá", "aquela", "aquele", "aquelas", "aqueles"
            # Adicione mais stop words conforme necessário
        ]

    def preprocess_text(self, df):
        """Preprocess the text data by cleaning and normalizing."""
        logger.info("Starting text preprocessing...")

        # Clean and normalize text in the 'page' column and create 'cleaned_text'
        df = df.withColumn("cleaned_text",
                           trim(
                               lower(
                                   regexp_replace(
                                       regexp_replace(
                                           regexp_replace(
                                               regexp_replace(
                                                   col("title"),
                                                   r'(https?:\/\/\S+)|(www\.\S+)', ''  # Remove URLs
                                               ),
                                               r'<[^>]+>', ''  # Remove tags HTML
                                           ),
                                           r'[^A-Za-z0-9ÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÇáéíóúàèìòùâêîôûãõç\s!"#$%&\'()*+,\-./:;<=>?@\[\]\\^_`{|}~]', ''  # Remove caracteres especiais
                                       ),
                                       r'\s+', ' '  # Substitui múltiplos espaços por um único espaço
                                   )
                               )
                           ) # Remove espaços em branco no início e final
        )

        logger.info("Text preprocessing completed.")
        return df

    def apply_tfidf(self, df):
        """Apply TF-IDF to the preprocessed text data."""
        logger.info("Applying TF-IDF...")

        # Tokenization
        tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="words")

        # Remove stopwords using the custom list
        remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=self.portuguese_stop_words)

        # HashingTF
        hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=300)

        # IDF
        idf = IDF(inputCol="rawFeatures", outputCol="features")

        # Create a pipeline
        pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])

        # Fit the pipeline to the data
        model = pipeline.fit(df)
        tfidf_df = model.transform(df)

        logger.info("TF-IDF application completed.")
        return tfidf_df

    def process(self, output_path):
        """Main method to process the TF-IDF."""
        logger.info("Starting TF-IDF processing...")

        # Load normalized items data
        items_df = self.spark.read.parquet(self.config.silver_path_itens_normalized)

        # Preprocess text
        preprocessed_df = self.preprocess_text(items_df)

        # Apply TF-IDF
        tfidf_df = self.apply_tfidf(preprocessed_df)

        # Process and save by year
        years = tfidf_df.select("year").distinct().collect()
        for year_row in years:
            year_val = year_row['year']
            logger.info(f"Processing year {year_val}...")

            year_df = tfidf_df.filter(col("year") == year_val)

            # Save by year
            logger.info(f"Saving data for year {year_val}...")
            (year_df
             .write
             .mode("append")
             .option("compression", "snappy")
             .partitionBy("year", "month", "day")
             .parquet(output_path))

            # Clear cache
            self.spark.catalog.clearCache()

        logger.info(f"TF-IDF features saved to {output_path}.")