import os
import sys
import numpy as np
import pandas as pd
import pickle

from src.INCOMEPREDICTION.logger import logging
from src.INCOMEPREDICTION.exception import CustomException

from sklearn.metrics import accuracy_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report ={}
        for i in range(len(models)):
            model=list(models.values())[i]
            #fit the train model
            model.fit(X_train,y_train)
            #predict value
            y_test_pred=model.predict(X_test)
            
            # find r2_score
            test_accuracy_score=accuracy_score(y_test,y_test_pred)
            report[list(models.keys())[i]]=test_accuracy_score
            
        return report
    except Exception as e:
        logging.info("Error osurred in model evaluate ")
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception occures at load_obj in function utils")
        raise CustomException(e,sys)
