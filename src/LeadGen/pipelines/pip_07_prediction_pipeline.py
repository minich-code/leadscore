import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from pathlib import Path
from src.LeadGen.logger import logger
from src.LeadGen.exception import CustomException
from src.LeadGen.utils.commons import read_yaml, load_object
from src.LeadGen.constants import *


@dataclass
class PredictionPipelineConfig:
    preprocessor_path: Path
    model_path: str
    model_name: str
    #Model parameters
    batch_size: int
    learning_rate: float
    epochs: int
    dropout_rates: dict
    optimizer: str
    loss_function: str
    activation_function: str

class ConfigurationManager:
    def __init__(self,
                 prediction_pipeline_config=PREDICTION_PIPELINE_FILEPATH,
                 params_config=PARAMS_CONFIG_FILEPATH):
        self.params = read_yaml(params_config)
        self.prediction_config = read_yaml(prediction_pipeline_config)

    def get_prediction_pipeline_config(self) -> PredictionPipelineConfig:
        pred_pipeline = self.prediction_config.prediction_pipeline
        params = self.params.dnn_params

        return PredictionPipelineConfig(
            preprocessor_path=Path(pred_pipeline.preprocessor_path),
            model_path=Path(pred_pipeline.model_path.format(model_name=pred_pipeline.model_name, epochs=params["epochs"])),
            model_name=Path(pred_pipeline.model_name),
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            dropout_rates=params['dropout_rates'],
            optimizer=params['optimizer'],
            loss_function=params['loss_function'],
            activation_function=params['activation_function']
        )

# Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self, input_dim, dropout_rates):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.dropout1 = nn.Dropout(p=dropout_rates['layer_1'])
        self.fc2 = nn.Linear(16, 8)
        self.dropout2 = nn.Dropout(p=dropout_rates['layer_2'])
        self.fc4 = nn.Linear(8, 1)
        self.dropout3 = nn.Dropout(p=dropout_rates['layer_3'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x

# Define the prediction pipeline class
class PredictionPipeline:
    def __init__(self, config: PredictionPipelineConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_preprocessor(self):
        try:
            logger.info("Loading preprocessor")
            preprocessor = load_object(self.config.preprocessor_path)
            return preprocessor
        except FileNotFoundError as e:
            logger.exception(f"Preprocessor file not found: {e}")
            raise CustomException(f"Error during loading preprocessor: {e}")
        except Exception as e:
            logger.exception("Failed to load preprocessor.")
            raise CustomException(f"Error during loading preprocessor: {e}")

    def load_model(self, input_dim):
        try:
            logger.info("Loading model")
            model = SimpleNN(input_dim=input_dim, dropout_rates=self.config.dropout_rates)
            model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        except FileNotFoundError as e:
            logger.exception(f"Model file not found: {e}")
            raise CustomException(f"Error during loading model: {e}")
        except Exception as e:
            logger.exception("Failed to load model.")
            raise CustomException(f"Error during loading model: {e}")

    def make_predictions(self, features):
        try:
            logger.info("Making Predictions")

            preprocessor = self.load_preprocessor()
            transformed_features = preprocessor.transform(features)

            model = self.load_model(input_dim=transformed_features.shape[1])
            inputs = torch.tensor(transformed_features, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                predictions = model(inputs).squeeze().cpu().numpy()

            logger.info("Predictions made successfully")
            return predictions
        except Exception as e:
            logger.exception("Failed to make predictions.")
            raise CustomException(f"Error during making predictions: {e}")


# Create a class to represent the input features
class CustomData:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_data_as_dataframe(self):
        try:
            logger.info("Converting data object to a dataframe")
            data_dict = {key: [getattr(self, key)] for key in vars(self)}
            return pd.DataFrame(data_dict)
        except Exception as e:
            logger.exception("Failed to convert data object to a dataframe.")
            raise CustomException(f"Error during converting data object to dataframe: {e}")