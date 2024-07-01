# test_technique_mailinblack

## Project Overview

This project aims to detect phishing URLs using machine learning models. It includes a complete pipeline for data preprocessing, feature engineering, model training, and prediction. The project also offers a web application for ease of interaction, including a React-based frontend and a FastAPI backend.

You can test the project in three ways:

- Using the notebook
- By running the frontend React server using npm and the FastAPI backend using Uvicorn
- By using Docker Compose

## Setup Instructions

### Prerequisites

1. Ensure you have `conda` installed.
2. Install `node` to run the React frontend.

### Environment Setup

#### 1. Create and Activate Conda Environment

```
conda create -n test python=3.11 -y
conda activate test
```

#### 2. Install Required Packages

```
pip install -r requirements.txt
```

### Running the Code

You can run the project in two ways:

#### 2.1 Through the Jupyter Notebook

1. Open the notebook and run all cells sequentially.
2. During the process, additional features are created that may take time. To save time, preprocessed data is provided and can be used directly.

#### 2.2 Through the Web Application

##### Frontend Setup

1. Navigate to the `frontend` directory:

   ```
   cd frontend
   ```
2. Install node modules:

   ```
   npm install
   ```
3. Start the React frontend server:

   ```
   npm start
   ```

##### Backend Setup

1. Ensure the conda environment is activated. If Not, run the following command:

   ```
   conda activate test

   ```
2. Ensure all required packages are installed. If Not, ruun the following command::

   ```
   pip install -r requirements.txt
   ```
3. Run the FastAPI backend server:

   ```
   uvicorn app:app --reload
   ```

#### 2.3 Through Docker

##### Docker Setup

1. Ensure Docker is installed and running on your system.
2. Build and run the Docker containers:

   ```
   docker-compose up --build
   ```

### Using the Web Application

The web application has two main functionalities: Training and Inference.

#### 3.1 Training

For training, you need to provide the following:

* **File Path** : Path to the data file.
* **Processed Data Checkbox** : Check if using preprocessed data. Recommended to avoid waiting too much for feature Engineering.
* **Stacking Checkbox** : Check if using stacking (combining multiple models and then training a meta model on top).
* **Search Type** : Select hyperparameter optimization method. `default` is recommended to avoid waiting too much time.

##### Recommended Parameters for Quick Testing

Keep all settings as default. To have a smooth testing and not waste too much time.

#### 3.2 Inference

For inference, provide the URL and choose the model and stacking options.

test_technique_mailinblack

## Project Structure

The project is structured into various modules, each responsible for a specific part of the pipeline:

* `solution.ipynb`: The notebook solution for the test case.
* `frontend`: Contains the frontend code.
  * `components`: Contains reusable UI components.
  * `pages`: Contains the main pages of the application:
    * `home`: The homepage.
    * `training`: The training page.
    * `inference`: The inference page.
* `src`: Contains the backend source code.
  * `components`: Contains base components for the backend:
    * `constants.py`: Contains constants used across the application.
    * `data_processing.py`: Responsible for loading, preprocessing, and saving data.
    * `feature_engineering.py`: Responsible for creating features from the URL.
    * `inference.py`: Contains the logic for making predictions using trained models.
    * `models.py`: Contains logic for model training and hyperparameter tuning.
    * `train.py`: Manages the training and evaluation of models.
  * `pipelines`: Contains the training pipeline:
    * `training_pipeline.py`: Defines the training pipeline.
  * `utils`: Contains helper functions.
* `app.py`: Contains the FastAPI application for training and prediction.
* `artifacts`: The folder where artifacts are stored.
  * `data`:
    * `raw`: The raw data is stored here.
    * `processed`: The processed data after analysis and feature engineering is done.
  * `models`: Where model artifacts are stored for inference.
* `logs`: Where the logs will be saved when running the notebook.
