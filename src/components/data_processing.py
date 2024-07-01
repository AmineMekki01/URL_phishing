"""
Module : data_processing.py
This module contains the DataProcessing class which is responsible for loading, preprocessing, and saving the data.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.dataframe import DataFrame

from src import logger

class DataProcessing:
    def __init__(self):
        self.spark = SparkSession.builder \
        .appName('PhishingURLDetection') \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.storage.memoryFraction", "0.6") \
        .getOrCreate()
    
    def load_data(self, file_path: str) -> DataFrame:
        """
        Load the data from the given file path

        Args:
            file_path (str): The path to the file

        Returns:
            DataFrame: The loaded data
        """
        try:
            df_spark = self.spark.read.csv(file_path, sep="\t", header=True, inferSchema=True)
            logger.info(f"Data loaded from {file_path}")
            return df_spark
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self, df_spark: DataFrame) -> DataFrame:
        """ Preprocess the data
        
        Args:
            df_spark (DataFrame): The input data
        
        Returns:
            DataFrame: The preprocessed data
        """
        try:
            df_spark = df_spark.withColumn('label', col('label').cast('integer'))
            df_spark = df_spark.drop('_c0')
            logger.info("Data preprocessed")
            return df_spark
        except Exception as e:
            logger.error(f"Error in preprocessing data: {e}")
            raise


    def compute_class_distribution(self, df_spark: DataFrame) -> DataFrame:
        """ Compute the class distribution
        
        Args:
            df_spark (DataFrame): The input data
        
        Returns:
            DataFrame: The class distribution
        """
        try:
            class_distribution = df_spark.groupBy('label').count().toPandas()
            logger.info("Class distribution computed")
            return class_distribution
        except Exception as e:
            logger.error(f"Error computing class distribution: {e}")
            raise

    def save_model(self, model, model_path: str) -> None:
        """
        Save the trained model to the specified path.

        Args:
            model (object): The trained model to be saved.
            model_path (str): The path where the model should be saved.

        Raises:
            Exception: If an error occurs while saving the model.
        """
        try:
            model.save(model_path)
            logger.info(f"Model saved at: {model_path}")
        except Exception as e:
            logger.error(f"Error in save_model: {e}")
            raise

        
