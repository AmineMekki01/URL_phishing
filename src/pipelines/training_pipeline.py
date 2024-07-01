from pyspark.sql import SparkSession
from src.components.data_processing import DataProcessing
from src.components.train import Trainer
from src.components.feature_engineering import FeatureEngineering
from pyspark.ml.feature import VectorAssembler
from src.components.constants import feature_cols

from src import logger

from pyspark.sql.functions import udf, col, length, regexp_replace


def execute_pipeline(file_path: str, model_path: str, model_name: str, search_type='default', user_params=None, is_processed=False, is_stacking=False, base_model_names=None, meta_model_name=None) -> dict:
    try:
        spark = SparkSession.builder \
            .appName("URLAnalysis") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.debug.maxToStringFields", "1000") \
            .getOrCreate()
        
        data_processing = DataProcessing()
        if is_processed:
            if file_path.endswith(".parquet"):
                df_spark = spark.read.parquet(file_path)
            elif file_path.endswith(".csv"):
                df_spark = spark.read.csv(file_path, header=True, inferSchema=True)
            else:
                raise Exception("Unsupported file format for processed data")
        else:
            df_spark = data_processing.load_data(file_path)
            df_spark = data_processing.preprocess_data(df_spark)
            feature_engineering = FeatureEngineering(spark=spark)
            df_spark = feature_engineering.create_features(df_spark)
            df_spark = feature_engineering.add_dns_features(df_spark)

        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_model')
        df_spark = assembler.transform(df_spark)
        
        train_data, test_data = df_spark.randomSplit([0.8, 0.2], seed=42)
        logger.info(f"user_params : {user_params}")
        try:
            trainer = Trainer(model_name, search_type, user_params, is_stacking, base_model_names, meta_model_name)
        except Exception as e:
            logger.error(f"Error in initializing the trainer : {e}")
            raise
        model = None
        base_models = None
        meta_model = None
        if is_stacking:
            base_models, meta_model = trainer.train_model(train_data)
        else:
            try:
                model = trainer.train_model(train_data)
            except Exception as e:
                logger.error(f"Error in training the model: {e}")
                raise
        
        accuracy, precision, recall, f1, auc = trainer.evaluate_model(test_data, base_models, meta_model, model)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }
    except Exception as e:
        logger.error(f"Error in execute_pipeline: {e}")
        raise