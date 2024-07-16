
from dataclasses import dataclass
from pathlib import Path


# Data Ingestion Entity 

@dataclass
class DataIngestionConfig:
    root_dir: Path
    mongo_uri: str
    database_name: str
    collection_name: str
    batch_size: int


# Data validation Entity 

@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    data_dir: Path
    all_schema: dict
    critical_columns: list  
    data_ranges: dict


# Data Transformation Entity

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    numerical_cols: list
    categorical_cols: list