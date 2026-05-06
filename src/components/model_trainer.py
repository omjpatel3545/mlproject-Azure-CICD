# Importing libraries

import os
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utilis import save_object, evaluate_models

@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test,y_test = (
                train_array[:,:-1], # all rows, all col except last one
                train_array[:,-1], # all rows, and just last col
                test_array[:,:-1], # all rows, all cols except last one
                test_array[:,-1], # all rows, and just last col
            )

            models = {
                'Linear Regression' : LinearRegression(),
                'Gradient Boosting' : GradientBoostingRegressor(),
                'K-Nearest Regressor' : KNeighborsRegressor(),
                'Decision Tree' : DecisionTreeRegressor(),
                'Random Forest Regressor' : RandomForestRegressor(),
                'XGBRegressor' : XGBRegressor(),
                'CatBoosting Regressor' : CatBoostRegressor(verbose=0),
                "AdaBoostRegressor" : AdaBoostRegressor()   
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test = X_test,y_test =y_test,models=models)

            # To get best model score from dict
            best_model_score = max(model_report.values())

            # To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            # Saving this model obj into pkl

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test ,predicted)
            return best_model_name, r2_square
        except Exception as e:
            raise CustomException(e,sys)

