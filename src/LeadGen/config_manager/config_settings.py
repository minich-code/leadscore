# File: data_ingestion.py

from src.LeadGen.utils.commons import read_yaml, create_directories
from src.LeadGen.constants import * 
from src.LeadGen.logger import logger
from src.LeadGen.config_entity.config_params import (DataIngestionConfig, DataValidationConfig)


class ConfigurationManager:
    def __init__(
        self, 
        data_ingestion_config=DATA_INGESTION_CONFIG_FILEPATH,
        data_validation_config=DATA_VALIDATION_CONFIG_FILEPATH, 
        schema_config=SCHEMA_CONFIG_FILEPATH
    ): 

        self.ingestion_config = read_yaml(data_ingestion_config)
        self.data_val_config = read_yaml(data_validation_config)
        self.schema = read_yaml(schema_config)

        create_directories([self.ingestion_config.artifacts_root])
        create_directories([self.data_val_config.artifacts_root])


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
    
# Data validation configuration manager 

    def get_data_validation_config(self) -> DataValidationConfig:
        data_valid_config = self.data_val_config.data_validation
        schema = self.schema.COLUMNS
        create_directories([data_valid_config.root_dir])
        logger.debug("Data validation configuration loaded")
        
        return DataValidationConfig(
            root_dir=data_valid_config.root_dir,
            STATUS_FILE=data_valid_config.STATUS_FILE,
            data_dir=data_valid_config.data_dir,
            all_schema=schema,
            critical_columns=data_valid_config.critical_columns,
            data_ranges=data_valid_config.data_ranges
        )