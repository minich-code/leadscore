# File: data_ingestion.py

from src.LeadGen.utils.commons import read_yaml, create_directories
from src.LeadGen.constants import * 
from src.LeadGen.config_entity.config_params import (DataIngestionConfig)


class ConfigurationManager:
    def __init__(
        self, 
        data_ingestion_config=DATA_INGESTION_CONFIG_FILEPATH
    ): 

        self.ingestion_config = read_yaml(data_ingestion_config)

        create_directories([self.ingestion_config.artifacts_root])


# Data ingestion config

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        data_config = self.ingestion_config.data_ingestion
        create_directories([data_config.root_dir])
        
        return DataIngestionConfig(
            root_dir=data_config.root_dir,
            mongo_uri=data_config.mongo_uri,
            database_name=data_config.database_name,
            collection_name=data_config.collection_name,
            batch_size=data_config.get('batch_size', 3000)
        )
