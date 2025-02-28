from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, explode, split, year, month, dayofmonth, trim
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline
from utils.logger import logger
from configs.config import Config


class TFIDFProcessor:
    def __init__(self, spark: SparkSession):
        """
        Initialize the TFIDFProcessor.

        Args:
            spark (SparkSession): A Spark session for processing data.

        Returns:
            None
        """
        self.spark = spark
        self.config = Config()


        self.portuguese_stop_words = [
            "a", "o", "as", "os", "um", "uma", "para", "com", "de", "do", "da", "em", "no", "na",
            "e", "que", "é", "do", "dos", "das", "se", "por", "como", "mas", "mais", "ou", "se",
            "não", "já", "tudo", "todos", "todas", "nada", "algum", "alguma", "alguns", "algumas",
            "quem", "onde", "quando", "como", "porque", "pelo", "pela", "pelo", "pelo", "pelo",
            "meu", "minha", "teu", "tua", "seu", "sua", "nosso", "nossa", "deles", "delas", "isso",
            "aquilo", "isto", "aqui", "ali", "lá", "aquela", "aquele", "aquelas", "aqueles"
     
        ]

    def preprocess_text(self, df):
        """Preprocess the text data by cleaning and normalizing.

        This method cleans the text data by removing URLs, HTML tags, and non-alphanumeric characters,
        converting the text to lowercase, and trimming whitespace.

        Args:
            df (DataFrame): The DataFrame containing the text data to preprocess.

        Returns:
            DataFrame: A DataFrame with an additional column 'cleaned_text' containing the preprocessed text.
        """
        logger.info("Starting text preprocessing...")

   
        df = df.withColumn("cleaned_text",
                           trim(
                               lower(
                                   regexp_replace(
                                       regexp_replace(
                                           regexp_replace(
                                               regexp_replace(
                                                   col("title"),
                                                   r'(https?:\/\/\S+)|(www\.\S+)', '' 
                                               ),
                                               r'<[^>]+>', ''  
                                           ),
                                           r'[^A-Za-z0-9ÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÇáéíóúàèìòùâêîôûãõç\s!"#$%&\'()*+,\-./:;<=>?@\[\]\\^_`{|}~]', ''  
                                       ),
                                       r'\s+', ' ' 
                                   )
                               )
                           ) 
        )

        logger.info("Text preprocessing completed.")
        return df

    def apply_tfidf(self, df):
        """Apply TF-IDF to the preprocessed text data.

        This method tokenizes the cleaned text, removes stop words, and computes the TF-IDF features.

        Args:
            df (DataFrame): The DataFrame containing the preprocessed text data.

        Returns:
            DataFrame: A DataFrame containing the TF-IDF features.
        """
        logger.info("Applying TF-IDF...")

      
        tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="words")

       
        remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=self.portuguese_stop_words)

   
        hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=300)

      
        idf = IDF(inputCol="rawFeatures", outputCol="features")

      
        pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])

      
        model = pipeline.fit(df)
        tfidf_df = model.transform(df)

        logger.info("TF-IDF application completed.")
        return tfidf_df

    def process(self, output_path):
        """Main method to process the TF-IDF.

        This method reads the normalized items data, preprocesses the text, applies TF-IDF,
        and saves the results partitioned by year, month, and day.

        Args:
            output_path (str): The path where the processed TF-IDF features will be saved.

        Returns:
            None
        """
        logger.info("Starting TF-IDF processing...")

    
        items_df = self.spark.read.parquet(self.config.silver_path_itens_normalized)

     
        preprocessed_df = self.preprocess_text(items_df)

     
        tfidf_df = self.apply_tfidf(preprocessed_df)

        
        years = tfidf_df.select("year").distinct().collect()
        for year_row in years:
            year_val = year_row['year']
            logger.info(f"Processing year {year_val}...")

            year_df = tfidf_df.filter(col("year") == year_val)

          
            logger.info(f"Saving data for year {year_val}...")
            (year_df
             .write
             .mode("append")
             .option("compression", "snappy")
             .partitionBy("year", "month", "day")
             .parquet(output_path))

        
            self.spark.catalog.clearCache()

        logger.info(f"TF-IDF features saved to {output_path}.")