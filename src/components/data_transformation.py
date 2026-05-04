import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import saved_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this function is respolsible for data transformation
        '''
        try:
            numerical_columns = ['writing score','reading score']
            categorical_columns = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course',
            ]

            num_pipeline  = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    # ('scaler',StandardScaler())
                ]
            )

            logging.info("Numerical columns Standard Scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test completed")
            logging.info("Obtraining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            trarget_column_name = 'math score'
            numerical_columns = ['writing score','reading score']


            input_feature_train_df = train_df.drop(columns=[trarget_column_name],axis=1)
            trarget_feature_train_df = train_df[trarget_column_name]

            input_feature_test_df = test_df.drop(columns=[trarget_column_name],axis=1)
            trarget_feature_test_df = test_df[trarget_column_name]

            logging.info(f'Applying preprocessing object on training dataframe and testing dataframe.')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(trarget_feature_train_df)
            ]
        

            test_arr = np.c_[
                input_feature_test_arr, np.array(trarget_feature_test_df)
            ]

            logging.info(f'Saved preprocessing object.')

            saved_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path, 
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)