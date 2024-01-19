import os
import sys
import pandas as pd
import numpy as np

from src.INCOMEPREDICTION.logger import logging
from src.INCOMEPREDICTION.exception import CustomException

from sklearn.model_selection import train_test_split


class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_dataingestion(self):
        logging.info("data ingestion started")
        
        try:
            data=pd.read_csv(os.path.join("notebooks\data\Adult.csv"))
            logging.info("Read the dataset as data")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("saved data as raw.csv in artifacts")
            
            logging.info("Here am performing train_test_split")
            train_data,test_data=train_test_split(data,test_size=0.30)
            
            train_data.to_csv(self.ingestion_config.train_data_path)
            test_data.to_csv(self.ingestion_config.test_data_path)
            logging.info("Dataingestion part completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Error occured in initiate dataingestion")
            raise CustomException(e,sys)