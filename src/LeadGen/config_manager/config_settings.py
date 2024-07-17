from src.LeadGen.utils.commons import read_yaml, create_directories
from src.LeadGen.constants import * 
from src.LeadGen.logger import logger
from src.LeadGen.config_entity.config_params import * #(DataIngestionConfig, DataValidationConfig, DataTransformationConfig)


class ConfigurationManager:
    def __init__(
        self, 
        data_ingestion_config=DATA_INGESTION_CONFIG_FILEPATH,
        data_validation_config=DATA_VALIDATION_CONFIG_FILEPATH, 
        schema_config=SCHEMA_CONFIG_FILEPATH,
        data_preprocessing_config=DATA_TRANSFORMATION_FILEPATH,
        model_training_config=MODEL_TRAINER_CONFIG_FILEPATH, 
        params_config=PARAMS_CONFIG_FILEPATH,
        model_evaluation_config=MODEL_EVALUATION_CONFIG_FILEPATH, 
        model_validation_config=MODEL_VALIDATION_CONFIG_FILEPATH
    ): 

        self.ingestion_config = read_yaml(data_ingestion_config)
        self.data_val_config = read_yaml(data_validation_config)
        self.schema = read_yaml(schema_config)
        self.preprocessing_config = read_yaml(data_preprocessing_config)
        self.training_config = read_yaml(model_training_config)
        self.params = read_yaml(params_config)
        self.evaluation_config = read_yaml(model_evaluation_config)
        self.params = read_yaml(params_config)
        self.validation_config = read_yaml(model_validation_config)
        self.params = read_yaml(params_config)
        

        create_directories([self.ingestion_config.artifacts_root])
        create_directories([self.data_val_config.artifacts_root])
        create_directories([self.preprocessing_config.artifacts_root])
        create_directories([self.training_config.artifacts_root])
        create_directories([self.evaluation_config.artifacts_root])
        create_directories([self.validation_config.artifacts_root])


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

# Data Transformation      
    def get_data_transformation_config(self) -> DataTransformationConfig:
        transformation_config = self.preprocessing_config.data_transformation
        create_directories([transformation_config.root_dir])
        return DataTransformationConfig(
            root_dir=Path(transformation_config.root_dir),
            data_path=Path(transformation_config.data_path),
            numerical_cols=transformation_config.numerical_cols,
            categorical_cols=transformation_config.categorical_cols
        )
    

# Model training configuration manager     
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        trainer_config = self.training_config.model_trainer
        params = self.params.dnn_params
        create_directories([trainer_config.root_dir])
        return ModelTrainerConfig(
            root_dir=Path(trainer_config.root_dir),
            model_name=trainer_config.model_name,
            train_features_path=trainer_config.train_features_path,
            train_targets_path=trainer_config.train_targets_path,
            val_features_path=trainer_config.val_features_path,
            val_targets_path=trainer_config.val_targets_path,
            val_metrics_path=Path(trainer_config.val_metrics_path),
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            dropout_rates=params['dropout_rates'],
            optimizer=params['optimizer'],
            loss_function=params['loss_function'],
            activation_function=params['activation_function'],
            #mlflow 
            mlflow_uri=trainer_config.mlflow_uri
        )
    

# Model Evaluation    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        eval_config = self.evaluation_config.model_evaluation
        params = self.params.dnn_params
        create_directories([eval_config.root_dir])
        return ModelEvaluationConfig(
            root_dir=eval_config.root_dir,
            val_features_path=Path(eval_config.val_features_path),
            val_target_path=Path(eval_config.val_target_path),
            model_path=Path(eval_config.model_path.format(model_name=eval_config.model_name, epochs=params["epochs"])),
            metric_file_name=Path(eval_config.metric_file_name),
            validation_metrics_path=Path(eval_config.validation_metrics_path),
            training_metrics_path=Path(eval_config.training_metrics_path),
            report_path=Path(eval_config.report_path),
            confusion_matrix_report=Path(eval_config.confusion_matrix_report),
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            dropout_rates=params["dropout_rates"],
            optimizer=params["optimizer"],
            loss_function=params["loss_function"],
            activation_function=params["activation_function"]
        )

# Model validation        
    def get_model_validation_config(self) -> ModelValidationConfig:
        val_config = self.validation_config.model_validation
        params = self.params.dnn_params
        create_directories([val_config.root_dir])
        return ModelValidationConfig(
            root_dir=Path(val_config.root_dir),
            test_feature_path=Path(val_config.test_feature_path),
            test_target_path=Path(val_config.test_target_path),
            model_path=Path(val_config.model_path.format(model_name=val_config.model_name, epochs=params["epochs"])),
            class_report=Path(val_config.class_report),
            val_metric_file_name=Path(val_config.val_metric_file_name),
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            dropout_rates=params["dropout_rates"],
            optimizer=params["optimizer"],
            loss_function=params["loss_function"],
            activation_function=params["activation_function"]
        )


