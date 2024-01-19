import os
import sys

from src.INCOMEPREDICTION.utils.utils import load_object

from sklearn.metrics import accuracy_score
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import numpy as np



class ModelEvaluation:
    def __init__(self):
        pass
    
    def eval_matrics(self,actual,pred):
        acc_score=accuracy_score(actual,pred)
        
        return acc_score
    
    def initiate_modelevaluation(self,train_array,test_array):
        try:
            X_test,y_test=(test_array[:,:-1], test_array[:,-1])
            
            model_path=os.path.join("artifacts","model.pkl")
            
            model=load_object(model_path)
            
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            print(tracking_url_type_store)
            
            with mlflow.start_run():
                pred_y=model.predict(X_test)
                
                acc_score=self.eval_matrics(y_test,pred_y)
                
                mlflow.log_metric('accuracy_score',acc_score)
                
                if tracking_url_type_store!= "file":
                    
                    mlflow.sklearn.log_model(model,"model",registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model,"model")
                    
                    
        except Exception as e:
            raise e
                    
                
                
        