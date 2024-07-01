"""
Module : inference.py
Description : This module contains the Inference class which is responsible for making predictions using the trained models.
"""
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel, GBTClassificationModel, LogisticRegressionModel

from src.utils.common_function import get_model_path, get_meta_model_path
from src import logger

class Inference:
    def __init__(self, model_name=None, model_path=None, is_stacking=False, base_model_names=None, meta_model_name=None):
        self.model_name = model_name
        self.model_path = model_path
        self.is_stacking = is_stacking
        self.base_model_names = base_model_names
        self.meta_model_name = meta_model_name
        
        self.spark = SparkSession.builder \
            .appName('PhishingURLDetection') \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.storage.memoryFraction", "0.6") \
            .getOrCreate()


    def load_model(self, model_name: str, model_path: str) -> object:
        """
        Load the model from the given path
        
        Args:
            model_name (str): The name of the model
            model_path (str): The path to the model
            
        Returns:
            object: The loaded model
        """
        model_mapping = {
            'rf': RandomForestClassificationModel,
            'gbt': GBTClassificationModel,
            'lr': LogisticRegressionModel
        }
        
        model_class = model_mapping.get(model_name)
        if not model_class:
            raise ValueError('Model not supported')
        
        return model_class.read().load(model_path)

    def predict(self, df_spark: DataFrame) -> int:
        """
        Make predictions using the trained model
        
        Args:
            df_spark (DataFrame): The input data
        
        Returns:
            int: The prediction
        """
        try:
            predictions = self.model.transform(df_spark)
            prediction = predictions.select('prediction').collect()[0]['prediction']
            return prediction
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            raise
    
    def predict_stacking(self, df_spark: DataFrame) -> tuple:
        """
        Make predictions using the trained stacking model
        
        Args:
            df_spark (DataFrame): The input data
        
        Returns:
            tuple: The prediction and probability
        """
        try:
            original_features = df_spark.select('url', 'features_model')
            base_predictions = []
            
            for model_name in self.base_model_names:
                model_path = get_model_path(model_name) + "_for_stacking"
                model = self.load_model(model_name, model_path)
                model_predictions = model.transform(df_spark)
                prediction_col = f'{model_name}_prediction'
                model_predictions = model_predictions.withColumnRenamed('prediction', prediction_col)
                base_predictions.append(model_predictions.select('url', prediction_col))
            
            combined_predictions = original_features
            for preds in base_predictions:
                combined_predictions = combined_predictions.join(preds, on='url', how='inner')

            combined_predictions.show()

            meta_features = [f'{model_name}_prediction' for model_name in self.base_model_names] + ['features_model']
            assembler = VectorAssembler(inputCols=meta_features, outputCol='meta_features')
            combined_predictions = assembler.transform(combined_predictions)
            
            meta_model_path = get_meta_model_path(self.meta_model_name)
            meta_model = self.load_model(self.meta_model_name, meta_model_path)
            final_predictions = meta_model.transform(combined_predictions)
            final_predictions.show(truncate=False)        
            if final_predictions.count() > 0:
                prediction_row = final_predictions.select('prediction', 'probability').collect()[0]
                prediction = prediction_row['prediction']
                probability = prediction_row['probability'][int(prediction)] 
            else:
                raise ValueError("No predictions were made.")
            
            return prediction, probability
        except Exception as e:
            logger.error(f"Error in predict_stacking: {e}")
            raise