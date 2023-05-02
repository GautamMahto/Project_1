import pandas as pd
import numpy as np
import sys
import os
from Insurance.entity import config_entity
from Insurance.entity import artifact_entity
from Insurance.exception import InsuranceException
from Insurance import utils
from Insurance.logger import logging
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Export Collection data as pandas Dataframe")
            df:pd.DataFrame = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name
            )
            logging.info(f"save data in future store")

            # replace the null values
            df.replace(to_replace="NA", value=np.NAN, inplace=True)

            # save data for the future purpose
            logging.info(f"create a feature store dir if not available")
            feature_store_dir = os.path.dirname(
                self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            logging.info(f"save df to feature store folder")

            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,
                      index=False, header=True)

            train_df, test_df = train_test_split(
                df, test_size=self.data_ingestion_config.test_size, random_state=1)
            logging.info(
                "Spilliting Our data into the train and test datasets")

            logging.info("Creating a dataset directory if not available")
            dataset_dir = os.path.dirname(
                self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            logging.info("Saving Data into the feature store folder")
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path, index=False, header=True)

            # Prepare artifact folder
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )

        except Exception as e:
            raise InsuranceException(error_message=e, error_detail=sys)
