# common_function.py

from src.components.constants import models_path
from src import logger

def get_model_path(model_name: str) -> str:
    """
    Get the model path
    
    Args:
        model_name (str): The model name
    
    Returns:
        str: The model path
    """
    try:
        return models_path + model_name
    except Exception as e:
        logger.error(f"Error in get_model_path: {e}")
        raise

def get_meta_model_path(model_name: str) -> str:
    """
    Get the meta model path
    
    Args:
        model_name (str): The model name
    
    Returns:
        str: The meta model path
    """
    try:
        return models_path + "meta_model_" + model_name
    except Exception as e:
        logger.error(f"Error in get_meta_model_path: {e}")
        raise
