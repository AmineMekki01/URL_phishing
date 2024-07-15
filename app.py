"""
Module : app.py
Description : This module contains the FastAPI application for training and prediction of the models.
"""

import os
from typing import Optional, List, Dict
from pydantic import BaseModel
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel, GBTClassificationModel, LogisticRegressionModel
from src.components.feature_engineering import FeatureEngineering
from src.pipelines.training_pipeline import execute_pipeline
from src.components.constants import feature_cols
from src.components.inference import Inference
from src.utils.common_function import get_model_path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    file_path: str
    model_name: str
    search_type: str = 'default'
    user_params: Optional[Dict] = None
    is_processed: bool = False
    is_stacking: bool = False
    base_model_names: Optional[List[str]] = None
    meta_model_name: Optional[str] = None

class PredictRequest(BaseModel):
    is_stacking: bool = False
    base_model_names: Optional[List[str]] = None
    meta_model_name: Optional[str] = None
    model_name: str
    url: str

def load_model(model_name: str, model_path: str):
    """
    Load the model from the given path
    
    Args:
        model_name (str): The model name
        model_path (str): The model path    
    
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

@app.post("/train")
def train(request: TrainRequest) -> dict:
    """
    Train the model
    
    Args:
        request (TrainRequest): The request
    
    Returns:
        dict: The metrics
    """
    try:
        metrics = execute_pipeline(
            file_path=request.file_path,
            model_path=get_model_path(request.model_name),
            model_name=request.model_name,
            search_type=request.search_type,
            user_params=request.user_params,
            is_processed=request.is_processed,
            is_stacking=request.is_stacking,
            base_model_names=request.base_model_names,
            meta_model_name=request.meta_model_name
        )
        return {"status": "Training initiated", **metrics}
    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    """
    Predict the model
    
    Args:
        request (PredictRequest): The request
    
    Returns:
        dict: The prediction
    """
    try:
        inference = Inference(
            is_stacking=request.is_stacking, 
            base_model_names=request.base_model_names, 
            meta_model_name=request.meta_model_name
        )
        
        spark = SparkSession.builder.appName('PhishingURLInference').getOrCreate()
        
        data = {'url': [request.url]}
        df = pd.DataFrame(data)
        df_spark = spark.createDataFrame(df)

        feature_engineering = FeatureEngineering(spark=spark)
        df_spark = feature_engineering.create_features(df_spark)
        df_spark = feature_engineering.add_dns_features(df_spark)

        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_model')
        df_spark = assembler.transform(df_spark)

        if request.is_stacking:
            prediction, probability = inference.predict_stacking(df_spark)
        else:
            model_path = get_model_path(request.model_name)
            if not model_path or not os.path.exists(model_path):
                logger.error(f"Model path not supported or does not exist: {model_path}")
                raise HTTPException(status_code=400, detail="Model path not supported or does not exist")
            
            model = load_model(request.model_name, model_path)
            predictions = model.transform(df_spark)
            prediction_row = predictions.select('prediction', 'probability').collect()[0]
            prediction = prediction_row['prediction']
            probability = prediction_row['probability'][int(prediction)]
        
        return {"prediction": prediction, "probability": probability}
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
