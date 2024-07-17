import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from src.LeadGen.logger import logger
from src.LeadGen.exception import CustomException
from src.LeadGen.utils.commons import read_yaml, load_object
from src.LeadGen.constants import *

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
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_preprocessor(self):
        try:
            logger.info("Loading preprocessor")
            preprocessor_path = self.config['preprocessor_path']
            preprocessor = load_object(preprocessor_path)
            return preprocessor
        except Exception as e:
            logger.exception("Failed to load preprocessor.")
            raise CustomException(f"Error during loading preprocessor: {e}")

    def load_model(self, input_dim):
        try:
            logger.info("Loading model")
            model = SimpleNN(input_dim=input_dim, dropout_rates=self.config['dropout_rates'])
            model_path = self.config['model_path']
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.exception("Failed to load model.")
            raise CustomException(f"Error during loading model: {e}")

    def make_predictions(self, features):
        try:
            logger.info("Making Predictions")

            # Transform the features using preprocessor
            preprocessor = self.load_preprocessor()
            transformed_features = preprocessor.transform(features)

            # Load model and make predictions
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

# if __name__ == '__main__':
#     try:
#         # Example usage
#         # config_path = Path('config.yaml')  # Convert to Path object
#         # config = read_yaml(config_path)
#         prediction_pipeline = PredictionPipeline(config)

#         # Sample data
#         sample_data = CustomData(
#             Lead_Origin='API',
#             Lead_Source='Olark Chat',
#             Do_Not_Email='No',
#             Do_Not_Call='No',
#             TotalVisits=0,
#             Total_Time_Spent_on_Website=0,
#             Page_Views_Per_Visit=0,
#             Last_Activity='Page Visited on Website',
#             Specialization='Select',
#             How_did_you_hear_about_X_Education='Select',
#             What_is_your_current_occupation='Unemployed',
#             What_matters_most_to_you_in_choosing_a_course='Better Career Prospects',
#             Search='No',
#             Newspaper_Article='No',
#             X_Education_Forums='No',
#             Newspaper='No',
#             Digital_Advertisement='No',
#             Through_Recommendations='No',
#             Tags='Interested in other courses',
#             Lead_Quality='Low in Relevance',
#             Lead_Profile='Select',
#             City='Select',
#             Asymmetrique_Activity_Index='02.Medium',
#             Asymmetrique_Profile_Index='02.Medium',
#             Asymmetrique_Activity_Score=15,
#             Asymmetrique_Profile_Score=15,
#             A_free_copy_of_Mastering_The_Interview='No',
#             Last_Notable_Activity='Modified',
#             Country='Hong Kong'
#         )

#         features = sample_data.get_data_as_dataframe()
#         predictions = prediction_pipeline.make_predictions(features)
#         print(predictions)
#     except CustomException as e:
#         logger.error(f"Prediction pipeline failed: {e}")
