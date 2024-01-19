import os
import sys
import pandas as pd
 
from src.INCOMEPREDICTION.logger import logging
from src.INCOMEPREDICTION.exception import CustomException
from src.INCOMEPREDICTION.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(slf,feature):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            logging.info(f"Type of Input feature: {type(feature)}")
            logging.info(f"Shape of feature {feature.shape}")
            scaled_data=preprocessor.transform(feature)
            scaled_arr=scaled_data.toarray()
            prediction=model.predict(scaled_arr)
            
            return prediction

        except Exception as e:
            raise CustomException(e,sys)
        
class custom_data:
    def __init__(self,
                 age:float,
                 workclass:str,
                 fnlwgt:float,
                 education:str,
                 education_num:float,
                 marital_status:str,
                 occupation:str,
                 relationship:str,
                 capital_gain:float,
                 hours_per_week:float):
        self.age=age
        self.workclass=workclass
        self.fnlwgt=fnlwgt
        self.education=education
        self.education_num=education_num
        self.marital_status=marital_status
        self.occupation=occupation
        self.relationship=relationship
        self.capital_gain=capital_gain
        self.hours_per_week=hours_per_week
        
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                "age":[self.age],
                "workclass":[self.workclass],
                "fnlwgt":[self.fnlwgt],
                "education":[self.education],
                "education_num":[self.education_num],
                "marital_status":[ self.marital_status],
                "occupation":[self.occupation],
                "relationship":[self.relationship],
                "capital_gain":[self.capital_gain],
                "hours_per_week":[self.hours_per_week]
            }
            
            df=pd.DataFrame(custom_data_input_dict)
            logging.info("DatFrame collected")
            
            return df
        except Exception as e:
            logging.info("Error occured in prediction pipeline")
            raise CustomException(e,sys)
                        