from src.INCOMEPREDICTION.component.data_ingestion import DataIngestion
from src.INCOMEPREDICTION.component.data_transformation import DataTransformation
from src.INCOMEPREDICTION.component.model_trainer import ModelTrainer
from src.INCOMEPREDICTION.component.model_evaluation import ModelEvaluation

import os
import sys
import pandas as pd

from src.INCOMEPREDICTION.logger import logging
from src.INCOMEPREDICTION.exception import CustomException

obj=DataIngestion()
train_data_path,test_data_path=obj.initiate_dataingestion()

data_transformation=DataTransformation()
train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)

model_trainer=ModelTrainer()
model_trainer.initiate_model_training(train_arr,test_arr)


model_eval=ModelEvaluation()
model_eval.initiate_modelevaluation(train_arr,test_arr)
