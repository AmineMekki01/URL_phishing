"""
Module : train.py
Description : This module contains the Trainer class which is responsible for training the models and evaluating the performance.
"""
import os
import shutil
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame
from src.components.models import Models, StackingModels
from src import logger

class Trainer:

    def __init__(self, model_name, search_type='default', user_params=None, is_stacking=False, base_model_names=None, meta_model_name=None):
        self.model_name = model_name
        self.search_type = search_type
        self.user_params = user_params
        self.is_stacking = is_stacking
        self.base_model_names = base_model_names
        self.meta_model_name = meta_model_name
        self.model_instance = Models(model_name, search_type, user_params)
        self.model = None

    def train_model(self, train_data: DataFrame) -> Models:
        """
        Train the model with the given training data
        
        Args:
            train_data (DataFrame): The training data
        
        Returns:
            model: The trained model
        """
        try:
            if self.is_stacking:
                base_models, meta_model = self.train_stacking_model(train_data, self.base_model_names, self.meta_model_name)
                return base_models, meta_model
            else:
                if self.search_type == "grid":
                    self.model = self.model_instance.grid_search(train_data)
                elif self.search_type == "random":
                    self.model = self.model_instance.random_search(train_data)
                else:
                    self.model = self.model_instance.fit_model(train_data)
                
                self.model_instance.log_best_params(self.model)
                
                model_path = f"./artifacts/models/{self.model_name}"
                self.save_model(self.model, model_path)
                return self.model
            
            
        except Exception as e:
            logger.error(f"Error in train_model: {e}")
            raise
    
    def train_stacking_model(self, train_data, base_models, meta_model):
        try:
            stacking_model = StackingModels(self.base_model_names, self.meta_model_name)
            base_models, meta_model = stacking_model.fit_stacking(train_data)
            return base_models, meta_model
        except Exception as e:
            logger.error(f"Error in train_stacking_model: {e}")
            raise


    def evaluate_model(self, test_data: DataFrame, base_models: list = None, meta_model: object = None, model: object = None) -> tuple:
        """
        Evaluate the model with the given test data
        
        Args:
            test_data (DataFrame): The test data
            base_models (list): The list of base models for stacking
            meta_model (object): The meta model for stacking
            model (object): The trained model
        
        Returns:
            accuracy (float): The accuracy of the model
            precision (float): The precision of the model
            recall (float): The recall of the model
            f1 (float): The F1 score of the model
            auc (float): The AUC score of the model
        """
        try:
            if self.is_stacking:
                stacking_model = StackingModels(self.base_model_names, self.meta_model_name)
                predictions = stacking_model.predict(test_data, base_models, meta_model, self.base_model_names)
            else:
                predictions = model.transform(test_data)

            correct_predictions = predictions.filter(predictions.label == predictions.prediction).count()
            accuracy = correct_predictions / float(predictions.count())

            tp = predictions.filter((predictions.label == 1) & (predictions.prediction == 1)).count()
            fp = predictions.filter((predictions.label == 0) & (predictions.prediction == 1)).count()
            fn = predictions.filter((predictions.label == 1) & (predictions.prediction == 0)).count()

            precision = tp / float(tp + fp) if (tp + fp) != 0 else 0.0
            recall = tp / float(tp + fn) if (tp + fn) != 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0

            evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='rawPrediction')
            auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})

            return accuracy, precision, recall, f1, auc
        except Exception as e:
            logger.error(f"Error in evaluate_model: {e}")
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
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            
            model.write().overwrite().save(model_path)
            logger.info(f"Model saved at: {model_path}")
        except Exception as e:
            logger.error(f"Error in save_model: {e}")
            raise
