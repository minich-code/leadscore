from src.LeadGen.logger import logger
from src.LeadGen.pipelines.pip_01_data_ingestion import DataIngestionPipeline
# from src.LeadGen.pipelines.pip_02_data_validation import DataValidationPipeline
# from src.LeadGen.pipelines.pip_03_data_transformation import DataTransformationPipeline
# from src.LeadGen.pipelines.pip_04_model_trainer import ModelTrainerPipeline
# from src.LeadGen.pipelines.pip_05_model_evaluation import ModelEvaluationPipeline
# from src.LeadGen.pipelines.pip_06_model_validation import ModelValidationPipeline


COMPONENT_01_NAME = "DATA INGESTION COMPONENT"
try:
    logger.info(f"# ====================== {COMPONENT_01_NAME} Started! ============================== #")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.run()
    logger.info(f"# ====================== {COMPONENT_01_NAME} Terminated Successfully! ===============##\n\nx******************x")

except Exception as e:
    logger.exception(e)
    raise e
