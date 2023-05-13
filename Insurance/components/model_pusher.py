import pandas as pd
import numpy as np
import sys
import os
from Insurance.entity import config_entity
from Insurance.entity import artifact_entity
from Insurance.exception import InsuranceException
from Insurance import utils
from Insurance.utils import load_object,save_object
from Insurance.logger import logging
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from Insurance.predictor import ModelResolver
from Insurance.config import TARGET_COLUMN
from Insurance.entity.artifact_entity import ModelEvaluationArtifact,DataIngestionArtifact,DataTransformationArtifact,ModelPusherArtifact,ModelTrainerArtifact
from Insurance.entity.config_entity import ModelPusherConfig

class ModelPusher:
    def __init__(self,model_pusher_config:ModelPusherConfig,data_transforamtion_artifact:DataTransformationArtifact,model_trainer_artifact:ModelTrainerArtifact):
        try:
            self.model_pusher_config=model_pusher_config
            self.data_tranformation_artifact=data_transforamtion_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver=ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise InsuranceException(e,sys)
    
    def initiate_model_pusher(self,)-> ModelPusherArtifact:
        try:
            # mODEL AND TARGET encoder ko load karenge
            transformer=load_object(file_path=self.data_tranformation_artifact.transform_object_path)
            model=load_object(file_path=self.model_trainer_artifact.model_path)
            target_encoder=load_object(file_path=self.data_tranformation_artifact.target_encoder_path)
            
            # Model pusher dir
            save_object(file_path=self.model_pusher_config.pusher_transformer_path,obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path,obj=model)
            save_object(file_path=self.model_pusher_config.pusher_target_encoder_path,obj=target_encoder)

            # Save Model
            transformer_path=self.model_resolver.get_latest_save_transformer_path()
            model_path=self.model_resolver.get_latest_save_model_path()
            target_encoder_path=self.model_resolver.get_latest_save_target_encoder_path()

            save_object(file_path=transformer_path,obj=transformer)
            save_object(file_path=model_path,obj=model)
            save_object(file_path=target_encoder_path,obj=target_encoder)

            model_pusher_artifact=ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir,saved_model_dir=self.model_pusher_config.saved_model_dir)
            
            return model_pusher_artifact

        except Exception as e:
            raise InsuranceException(e,sys)
    