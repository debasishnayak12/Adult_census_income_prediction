import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.INCOMEPREDICTION.logger import logging
from src.INCOMEPREDICTION.exception import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

from src.INCOMEPREDICTION.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath=os.path.join("artifacts","preprocessor.pkl")
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation(self):
        
        try:
            logging.info("data transformation started")
            
            category_cols=['workclass', 'education', 'marital_status', 'occupation','relationship']
            
            numeric_cols=['age', 'fnlwgt', 'education_num', 'capital_gain','hours_per_week']
            
            logging.info("Pipeline initiated")
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer()),
                    ("scaler",StandardScaler())
                    ]
                )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                )
            
            preprocessor=ColumnTransformer(
                transformers=[
                    ('num_pipeline',num_pipeline,numeric_cols),
                    ('cat_pipeline',cat_pipeline,category_cols)
                    ])
            
            return preprocessor
        
        except Exception as e:
            logging.info("Error occured in get data transfomration")
            raise CustomException(e,sys)
        
        
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Shape of Train Dataframe Head : \n{train_df.shape}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessing_obj=self.get_data_transformation()
            
            target_column="salary"
            #I have checked in research.ipynb file that those column i mentioned in drop_columns are less contribute in model prediction
            drop_columns=['Unnamed: 0','salary','sex','race','capital_loss','country']
            category_cols=['workclass', 'education', 'marital_status', 'occupation','relationship']
            
            numeric_cols=['age', 'fnlwgt', 'education_num', 'capital_gain','hours_per_week']
            
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[[target_column]]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[[target_column]]
            
            input_feature_train_data=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_data=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info(f"Type of input_feature_train_data :{type(input_feature_train_data)}")
            logging.info(f"Shape of target_feature_train_df:{ target_feature_train_df.shape}")
            logging.info(f"Shape of target_feature_test_df: { target_feature_test_df.shape}")
            logging.info(f"Shape of input_feature_train_arr: { input_feature_train_data.shape}")
            
            # Get feature names from the ColumnTransformer
            feature_names = preprocessing_obj.named_transformers_['num_pipeline'].named_steps['scaler'].get_feature_names_out(numeric_cols).tolist() + \
                preprocessing_obj.named_transformers_['cat_pipeline'].named_steps['encoder'].get_feature_names_out(category_cols).tolist()

             
            train_data = pd.DataFrame(input_feature_train_data.toarray(), columns=feature_names)
            test_data = pd.DataFrame(input_feature_test_data.toarray(), columns=feature_names)
            
            train_arr = np.c_[np.array(train_data), np.array(target_feature_train_df)]
            test_arr = np.c_[np.array(test_data), np.array(target_feature_test_df)]

            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_filepath,
                obj=preprocessing_obj
            )
            
            logging.info("Preprocessor_obj file saved as preprocessor.pkl file")
            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            logging.info("Error occured in initialize data transoformation ")
            raise CustomException(e,sys)