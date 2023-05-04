from Insurance.logger import logging
from Insurance.exception import InsuranceException
import os,sys
from Insurance.utils import get_collection_as_dataframe
from Insurance.entity.config_entity import DataIngestionConfig
from Insurance.entity.config_entity import DataValidationConfig
from Insurance.entity import config_entity
from Insurance.components.data_ingestion import DataIngestion
from Insurance.components.data_validation import DataValidation

# def test_logger_and_exception():
#     try:
#         logging.info('Starting the test logger and exception file ')
#         result=3/0
#         print(result)
#         logging.info('Ending point of the test logger and exception file ')
#     except Exception as e:
#         logging.debug(str(e))
#         raise InsuranceException(e,sys)
    
if __name__== "__main__":
    try:
        #test_logger_and_exception()
        #get_collection_as_dataframe(database_name="INSURANCE",collection_name='INSURANCE_PROJECT')
        training_pipeline_config=config_entity.TrainingPipelineConfig()
        data_ingestion_config=config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())

        data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        
        # Data Validatition
        data_validation_config=config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation=DataValidation(data_validation_config=data_validation_config,data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact=data_validation.inititate_data_validation()

    except Exception as e:
        print(e)