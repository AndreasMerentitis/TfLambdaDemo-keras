try:
  import unzip_requirements
except ImportError:
  pass

import json
import os
import tarfile

import boto3
import tensorflow as tf
import numpy as np

import pandas as pd
from keras import models
from sklearn.preprocessing import LabelEncoder

import census_data

import logging

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

FILE_DIR = './'
S3_MODEL_DIR = ''

BUCKET = 'serverless-ml-1'

def _easy_input_function_keras(data_dict):
    transform_needed = [False,
               True,
               False,
               True,
               False,
               True,
               True,
               True,
               True,
               True,
               False,
               False,
               False,
               True,
               True]
    
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
       'marital_status', 'occupation', 'relationship', 'race', 'gender',
       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
               
    df = pd.DataFrame(data_dict.items())[1]
    full_data = pd.DataFrame.from_records(df.values)
    full_data = full_data.T
    data = np.zeros(shape=(full_data.shape[0], full_data.shape[1]), dtype=np.float32)

    for i in range(len(transform_needed)):
        print (i)
        if transform_needed[i]:
            tmp_data = full_data.iloc[:, i].tolist()
            encoder = LabelEncoder()
            encoder.fit(tmp_data)
            data[:, i] = encoder.transform(tmp_data)
        else:
            data[:, i] = full_data.iloc[:, i].tolist()


    x_predict = data
    return x_predict



def _easy_input_function_tf(data_dict, batch_size=64):
    """
    data_dict = {
        '<csv_col_1>': ['<first_pred_value>', '<second_pred_value>']
        '<csv_col_2>': ['<first_pred_value>', '<second_pred_value>']
        ...
    }
    """

    # Convert input data to numpy arrays
    for col in data_dict:
        col_ind = census_data._CSV_COLUMNS.index(col)
        dtype = type(census_data._CSV_COLUMN_DEFAULTS[col_ind][0])
        data_dict[col] = np.array(data_dict[col],
                                        dtype=dtype)

    labels = data_dict.pop('income_bracket')

    ds = tf.data.Dataset.from_tensor_slices((data_dict, labels))
    ds = ds.batch(64)

    return ds


def inferHandler():
    
    with open('example_body.json') as f:
        body = json.load(f)
    
    #logging.warning('body value is %s', body)

    # Read in prediction data as dictionary
    # Keys should match _CSV_COLUMNS, values should be lists
    predict_input = body['input']
    logging.warning('predict_input value is %s', predict_input)

    # Read in epoch
    epoch_files = body['epoch']
    logging.warning('epoch_files value is %s', epoch_files)
    
    s3_full_model_path = os.path.join(S3_MODEL_DIR,'model.tar.gz')
    logging.warning('model_path value is %s', s3_full_model_path)
    
    logging.warning('BUCKET value is %s', BUCKET)

    # Download model from S3 and extract
    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(s3_full_model_path,'model.tar.gz')

    tarfile.open(FILE_DIR+'model.tar.gz', 'r').extractall(FILE_DIR)

    # Create feature columns
    wide_cols, deep_cols = census_data.build_model_columns()
    
    custom_model_dir=os.path.join(FILE_DIR+'model_ML.h5')
    
    x_test = _easy_input_function_keras(predict_input)
    print (x_test)


    # make predictions from the h5 file after loading with keras
    h5_model = models.load_model('model_ML.h5')
    x_test_casted = np.reshape(x_test[0,:13],(13,1))
    h5_predictions = h5_model.predict(x_test_casted.T)
    print (h5_predictions)

    response = {
        "statusCode": 200,
        "body": json.dumps(h5_predictions.tolist(),
                            default=lambda x: x.decode('utf-8'))
    }
    
    logging.warning('response value is %s', response)

    return response
    
    
def main():
    print("Hello World!")
    inferHandler()


if __name__ == "__main__":
    main()



    

