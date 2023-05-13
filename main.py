from Insurance.logger import logging
from Insurance.exception import InsuranceException
import os,sys
from Insurance.utils import get_collection_as_dataframe
from Insurance.entity.config_entity import DataIngestionConfig
from Insurance.entity.config_entity import DataValidationConfig
from Insurance.entity.config_entity import DataTransformationConfig
from Insurance.entity.config_entity import ModelTrainingConfig
from Insurance.entity.config_entity import ModelEvaluationConfig
from Insurance.entity.config_entity import ModelPusherConfig
from Insurance.entity import config_entity
from Insurance.components.data_ingestion import DataIngestion
from Insurance.components.data_validation import DataValidation
from Insurance.components.data_transformation import DataTransformation
from Insurance.components.model_trainer import ModelTrainer
from Insurance.components.model_evaluation import ModelEvaluation,ModelResolver
from Insurance.components.model_pusher import ModelPusher


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

        # Data Transformation
        data_transformation_config=config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation=DataTransformation(data_transformation_config=data_transformation_config,data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact=data_transformation.initiate_data_transformation()

        # Model Trainer
        model_trainer_config=config_entity.ModelTrainingConfig(training_pipeline_config=training_pipeline_config)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.inititate_model_trainer()

        # Model Evaluation
        model_evaluation_config=config_entity.ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
        model_evaluation=ModelEvaluation(model_evaluation_config=model_evaluation_config,
                                        data_ingestion_artifact=data_ingestion_artifact,
                                        data_transformation_artifact=data_transformation_artifact,
                                        model_trainer_artifact=model_trainer_artifact)
        model_evaluation_artifact=model_evaluation.initiate_model_evaluation()


        # Model Pusher
        model_pusher_config=config_entity.ModelPusherConfig(training_pipeline_config=training_pipeline_config)
        model_pusher=ModelPusher(model_pusher_config=model_pusher_config,
                                 data_transforamtion_artifact=data_transformation_artifact,
                                 model_trainer_artifact=model_trainer_artifact)
        Model_pusher_artifact= model_pusher.initiate_model_pusher()

    except Exception as e:
        print(e)