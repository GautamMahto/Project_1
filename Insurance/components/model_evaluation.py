import pandas as pd
import numpy as np
import sys
import os
from Insurance.entity import config_entity
from Insurance.entity import artifact_entity
from Insurance.exception import InsuranceException
from Insurance import utils
from Insurance.utils import load_object
from Insurance.logger import logging
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from Insurance.predictor import ModelResolver
from Insurance.config import TARGET_COLUMN


class ModelEvaluation:
    def __init__(self,model_evaluation_config:config_entity.ModelTrainingConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact:artifact_entity.ModelTrainerArtifact):
        try:
            self.model_evaluation_artifact=model_evaluation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver=ModelResolver()

        except Exception as e:
            raise InsuranceException(e,sys)
    
    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            latest_dir_path=self.model_resolver.get_latest_dir_path()

            if latest_dir_path==None:
                model_evaluation_artifact=artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,improved_accuracy=None)
                logging.info(f"Model Evaluation Artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact
            
    
            # Find Location of Previous Model
            transformer_path=self.model_resolver.get_latest_transformer_path()
            model_path=self.model_resolver.get_latest_model_path()
            target_encoder_path=self.model_resolver.get_latest_target_encoder_path()

            # previous Model
            transformer=load_object(file_path=transformer_path)
            model=load_object(file_path=model_path)
            target_encoder=load_object(file_path=target_encoder_path)

            # Current Model
            current_transformer=load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model=load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder=load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df=test_df[TARGET_COLUMN]
            y_true=target_df

            input_features_name=list(transformer.feature_names_in_)
            for i in input_features_name:
                if test_df[i]=='O':
                    test_df[i]=target_encoder.fit_transform(test_df[i])
            
            input_arr=transformer.transform(test_df[i])
            y_pred=model.predict(input_arr)

            # Comparision between new modell and old model
            previous_model_score=r2_score(y_true=y_true,y_pred=y_pred)

            # Accuracy Current Model
            input_feature_name=list(transformer.feature_names_in_)
            input_arr=current_transformer(test_df[input_feature_name])
            y_pred=current_model.predict(input_arr)
            y_true=target_df

            current_model_score=r2_score(y_true=y_true,y_pred=y_pred)

            #final comparision between the model
            if current_model_score<=previous_model_score:
                logging.info(f"Current trained model is not better than previous one")
                raise Exception("Current model is not better than previous one")
            
            model_evaluation_artifact=artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,improved_accuracy=current_model_score-previous_model_score)

            return model_evaluation_artifact
    
        except Exception as e:
            raise InsuranceException(e,sys)