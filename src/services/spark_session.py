from pyspark.sql import SparkSession

class SparkSessionFactory:
    def create_spark_session(self, app_name: str):
        print("Initializing Spark Session...")
        spark = SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()
        print("Spark Session initialized.")
        return spark

