import os
import sys
import logging
from datetime import datetime

start_time = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok=True)

LOG_FILE = f"{start_time}.log"
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging_str = (
    "[ %(asctime)s ] PID: %(process)d | TID: %(thread)d | Module: %(module)s |" 
    "Function: %(funcName)s | Line: %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

logging.basicConfig(
    format=logging_str,
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("URLAnalysis")