import pandas as pd
import numpy as np
import sys
import os
from Insurance.entity import config_entity
from Insurance.entity import artifact_entity
from Insurance.exception import InsuranceException
from Insurance import utils
from Insurance.logger import logging
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# MOdel Define & Trainer


class ModelTrainer:
    def __init__(self,model_trainer_config:config_entity.ModelTrainingConfig,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise InsuranceException(e,sys)
    
    def train_model(self,x,y):
        try:
            lr=LinearRegression()
            lr.fit(x,y)
            return lr
        
        except Exception as e:
            raise InsuranceException(e,sys)
        
    def inititate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            train_arr=utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transform_train_path)
            test_arr=utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transform_test_path)

            x_train,y_train=train_arr[:, :-1],train_arr[:,-1]
            x_test,y_test=train_arr[:,:-1],train_arr[:,-1]

            model=self.train_model(x=x_train,y=y_train)

            yhat_train=model.predict(x_train)
            r2_train_score=r2_score(y_true=y_train,y_pred=yhat_train)

            yhat_test=model.predict(x_test)
            r2_test_score=r2_score(y_true=y_test,y_pred=yhat_test)

            if r2_test_score < self.model_trainer_config.expected_accuracy:
                raise Exception(f"Model is not got as it is not given\
                                 the expected acccuracy:{self.model_trainer_config.expected_accuracy},model actual score:{r2_test_score}")
            
            diff=abs(r2_train_score-r2_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train Model and test score diff:{diff} is more than overfitting threshold{self.model_trainer_config.overfitting_threshold}")

            utils.save_object(file_path=self.model_trainer_config.model_path,obj=model)

            model_trainer_artifact= artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path,r2_train_score=r2_train_score,r2_test_score=r2_test_score)

            return model_trainer_artifact

        except Exception as e:
            raise InsuranceException(e,sys)