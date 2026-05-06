import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utilis import save_object
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    '''
    It saves the constants that are used in data tranformation process. It might includes save paths, model name, dataset path
    '''
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTranformation:

    '''
    It mainly does 3 things
    1. Reads data
    2. Transforms data
    3. Saves the transformer
    '''

    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            numerical_columns = ["writing_score","reading_score"]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler()) # mean - 0 , std - 1
                ]
            )


            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("OHE",OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                    ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
        
            preprocessor = ColumnTransformer([
                ("num_pipelines",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
            ])
        
            return preprocessor # this is a ready to use transformer
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train/test data is completed.")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_featur_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_featur_test_array = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_featur_train_array,np.array(target_feature_train_df)
                ]

            test_arr = np.c_[
                input_featur_test_array,np.array(target_feature_test_df)
                ]

            logging.info(f"Saved preprocessing object")


            save_object(
                file_path = self.data_tranformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path # path of our transformer machine (preprocessor.pkl)
            )
        except Exception as e:
            raise CustomException(e,sys)
           


               
