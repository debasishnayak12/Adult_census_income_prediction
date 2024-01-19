import os
import sys
import pandas as pd
import numpy as np

from src.INCOMEPREDICTION.logger import logging
from src.INCOMEPREDICTION.exception import CustomException

from dataclasses import dataclass
from src.INCOMEPREDICTION.utils.utils import save_object
from src.INCOMEPREDICTION.utils.utils import evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

@dataclass
class ModelTrainerConfig:
    trianed_model_filepath=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
            "LogisticRegression":LogisticRegression(solver='liblinear'),
            "SVC":SVC(),
            "DecisionTreeClassifier":DecisionTreeClassifier(),
            "RandomForestClassifier":RandomForestClassifier()
        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            
            best_accuracy_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_accuracy_score)
            ]
            
            best_model = models[best_model_name]
            
            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy_Score : {best_accuracy_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy_Score : {best_accuracy_score}')
            
            save_object(
                 file_path=self.model_trainer_config.trianed_model_filepath,
                 obj=best_model
            )
            
        except Exception as e:
            logging.info('Error occured at ModelTraining')
            raise CustomException(e,sys)

