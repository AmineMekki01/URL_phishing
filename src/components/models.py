"""
Module : models.py
Description : This module contains the Models class which is responsible for training and hyperparameter tuning of the models.
"""
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame

from src import logger

class Models:

    def __init__(self, model_name, search_type='default', user_params=None, feature_cols='features_model'):
        self.model_name = model_name
        self.search_type = search_type
        self.user_params = user_params
        self.model = None
        self.feature_cols = feature_cols

    def initiate_model(self):
        if self.model_name == 'rf':
            self.model = RandomForestClassifier(labelCol='label', featuresCol=self.feature_cols)
        elif self.model_name == 'gbt':
            self.model = GBTClassifier(labelCol='label', featuresCol=self.feature_cols)
        elif self.model_name == 'lr':
            self.model = LogisticRegression(labelCol='label', featuresCol=self.feature_cols)
        else:
            raise ValueError('Model not supported')

        return self.model
        
    def fit_model(self, train_data: DataFrame) -> object:
        self.model = self.initiate_model()
        return self.model.fit(train_data)
        
    def grid_search(self, train_data: DataFrame) -> object:
        self.model = self.initiate_model()
        paramGrid = self.get_param_grid()
        evaluator = BinaryClassificationEvaluator(labelCol='label')
        crossval = CrossValidator(estimator=self.model,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=3)
        cv_model = crossval.fit(train_data)
        return cv_model

    def random_search(self, train_data: DataFrame) -> object:
        self.model = self.initiate_model()
        try:
            paramGrid = self.get_param_grid()
            evaluator = BinaryClassificationEvaluator(labelCol='label')
            trainValidationSplit = TrainValidationSplit(estimator=self.model,
                                                        estimatorParamMaps=paramGrid,
                                                        evaluator=evaluator,
                                                        trainRatio=0.8)
            tvs_model = trainValidationSplit.fit(train_data)
            return tvs_model
        except Exception as e:
            logger.info(f"Error in random search training : {e}")
     
    def get_param_grid(self):
        if self.search_type == 'default':
            paramGrid = self.ger_default_params()
        elif self.search_type == 'random' or self.search_type == 'grid':
            paramGrid = self.def_get_user_params()
        else:
            raise ValueError('Invalid search type')
        return paramGrid
     
    def def_get_user_params(self):
        paramGrid = ParamGridBuilder()
        if self.user_params:
            for param, values in self.user_params.items():
                if hasattr(self.model, param):
                    if isinstance(values, list):
                        try:
                            numeric_values = []
                            for value in values:
                                try:
                                    numeric_value = float(value)
                                    if numeric_value.is_integer():
                                        numeric_value = int(numeric_value)
                                    numeric_values.append(numeric_value)
                                except ValueError:
                                    numeric_values.append(value)
                            paramGrid = paramGrid.addGrid(getattr(self.model, param), numeric_values)
                        except ValueError:
                            raise ValueError(f"Invalid param value given for param {param}: {values}")
                    else:
                        raise ValueError(f"Param values for {param} should be a list")
        return paramGrid.build()

    def ger_default_params(self):
        try:
            if self.model_name == 'rf':
                paramGrid = (ParamGridBuilder()
                            .addGrid(self.model.numTrees, [50, 100, 150, 200])
                            .addGrid(self.model.maxDepth, [5, 10, 20, 30])
                            .addGrid(self.model.maxBins, [32, 64, 128])
                            .addGrid(self.model.minInstancesPerNode, [1, 2, 4])
                            .addGrid(self.model.subsamplingRate, [0.5, 0.7, 1.0])
                            .addGrid(self.model.featureSubsetStrategy, ['auto', 'sqrt', 'log2'])
                            .build())
            elif self.model_name == 'gbt':
                paramGrid = (ParamGridBuilder()
                            .addGrid(self.model.maxIter, [10, 20, 30, 50])
                            .addGrid(self.model.maxDepth, [3, 5, 7, 10])
                            .addGrid(self.model.maxBins, [32, 64, 128])
                            .addGrid(self.model.minInstancesPerNode, [1, 2, 4])
                            .addGrid(self.model.stepSize, [0.01, 0.1, 0.2])
                            .addGrid(self.model.subsamplingRate, [0.5, 0.7, 1.0])
                            .build())
            elif self.model_name == 'lr':
                paramGrid = (ParamGridBuilder()
                            .addGrid(self.model.regParam, [0.01, 0.1, 1.0, 10.0])
                            .addGrid(self.model.elasticNetParam, [0.0, 0.5, 1.0])
                            .addGrid(self.model.maxIter, [10, 50, 100])
                            .addGrid(self.model.fitIntercept, [True, False])
                            .addGrid(self.model.tol, [1e-6, 1e-4, 1e-2])
                            .build())
        except Exception as e:
            logger.info(f"Error in getting params : {e}")
        return paramGrid

    def log_best_params(self, trained_model):
        """
        Log the best hyperparameters found by the model training.

        Args:
            trained_model (object): The trained model object which contains the best parameters.

        Raises:
            Exception: If an error occurs while logging the best parameters.
        """
        try:
            if isinstance(trained_model, CrossValidator):
                best_model = trained_model.bestModel
                best_params = {param.name: best_model.getOrDefault(param) for param in best_model.extractParamMap()}
                logger.info(f"Best hyperparameters found: {best_params}")
            elif isinstance(trained_model, TrainValidationSplit):
                best_model = trained_model.bestModel
                best_params = {param.name: best_model.getOrDefault(param) for param in best_model.extractParamMap()}
                logger.info(f"Best hyperparameters found: {best_params}")
            else:
                logger.warning("The trained model does not have bestModel attribute")
        except Exception as e:
            logger.error(f"Error in logging best hyperparameters: {e}")
            raise
        

class StackingModels:
    def __init__(self, base_model_names, meta_model_name, search_type='default', user_params=None):
        self.base_models = [Models(name, search_type, user_params) for name in base_model_names]
        self.meta_model = Models(meta_model_name, search_type, user_params, feature_cols='meta_features')

    def fit_stacking(self, train_data : DataFrame) -> object:
        """
        Train the base models and meta-model for stacking
        
        Args:
            train_data (DataFrame): The training data
        
        Returns:
            base_models: List of trained base models
            meta_model: The trained meta-model
        """
        base_model_predictions = []
        base_models = []
        for model in self.base_models:
            base_model = model.fit_model(train_data)
            preds = base_model.transform(train_data)
            prediction_col = f'{model.model_name}_prediction'
            preds = preds.withColumnRenamed('prediction', prediction_col)
            base_model_predictions.append(preds.select('url', prediction_col, 'label'))
            logger.info(f'{model.model_name} trained and predictions obtained successfully')
            
            model_path = f"./artifacts/models/{model.model_name}_for_stacking"
            base_model.write().overwrite().save(model_path)
            logger.info(f"Model saved at: {model_path}")
            base_models.append(base_model)
        
        combined_predictions = base_model_predictions[0]
        for preds in base_model_predictions[1:]:
            combined_predictions = combined_predictions.join(preds, on=['url', 'label'])
        
        combined_predictions = combined_predictions.join(train_data.select('url', 'features_model'), on='url')

        meta_features = [f'{model.model_name}_prediction' for model in self.base_models] + ['features_model']
        assembler = VectorAssembler(inputCols=meta_features, outputCol='meta_features')
        combined_predictions = assembler.transform(combined_predictions)

        combined_predictions = combined_predictions.select('meta_features', 'label')

        meta_model = self.meta_model.fit_model(combined_predictions)
        model_path = f"./artifacts/models/meta_model_{self.meta_model.model_name}"
        meta_model.write().overwrite().save(model_path)
        logger.info(f"Meta model saved at: {model_path}")
        return base_models, meta_model

    def predict(self, test_data : DataFrame, base_models : list, meta_model : object, base_model_names : list) -> DataFrame:
        """
        Predict with the base models and meta-model
        
        Args:
            test_data (DataFrame): The test data
            base_models (list): List of trained base models
            meta_model: The trained meta-model
            base_model_names (list): List of base model names
        
        Returns:
            predictions: The predictions
        """
        base_model_predictions = []
        for model in base_models:
            preds = model.transform(test_data)
            base_model_name = base_model_names[base_models.index(model)]
            prediction_col = f'{base_model_name}_prediction'
            preds = preds.withColumnRenamed('prediction', prediction_col)
            base_model_predictions.append(preds.select('url', prediction_col, 'label'))
    
        combined_predictions = base_model_predictions[0]
        for preds in base_model_predictions[1:]:
            combined_predictions = combined_predictions.join(preds, on=['url', 'label'])

        combined_predictions = combined_predictions.join(test_data.select('url', 'features_model'), on='url')

        meta_features = [f'{model_name}_prediction' for model_name in base_model_names] + ['features_model']
        assembler = VectorAssembler(inputCols=meta_features, outputCol='meta_features')
        combined_predictions = assembler.transform(combined_predictions)
        predictions = meta_model.transform(combined_predictions)
        return predictions
    