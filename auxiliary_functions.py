import io
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#Statistical analysis of obtained datasets
#dataframe -    DataFrame to be used for the analysis
#filename -     Name of output file. Should be similar to dataset used to avoid confusion
#pred_col -     Column used for predictions
#pred_values -  List of all possible values for the prediction
def analysis(dataframe, filename = "analysis.txt", pred_col = None, pred_values = [1,0]):
    if isinstance(dataframe, pd.DataFrame):
        if pred_col is not None:
            pred_counts = []
            for pred_value in pred_values:
                print(pred_value)
                count = dataframe[pred_col].value_counts()[pred_value]
                #count = dataframe.groupby(pred_col).count()
                pred_counts.append((pred_value, count))
                print("Number of predictions resulting in a value of ", pred_value, ": ", count)

        data_info = io.StringIO()
        dataframe.info(buf=data_info)
        data_info = data_info.getvalue()
        data_desc = dataframe.describe(include = 'all').to_string()
        print("Non-null values present in each column\n", data_info)
        print("Statistical information of each column\n", data_desc)
        with open(filename, 'w') as file:
            file.write("---Data info---\n")
            file.write(data_info)
            file.write("\n---Data Description---\n")
            file.write(data_desc)
            if pred_col is not None:
                file.write("\n---Prediction column values---\n")
                file.write(str(pred_counts))
    else:
        print("Given dataframe is not of type pandas.DataFrame")


# Prepare the data for prediction and explanation. It is suggested to use the last two parameters in order
# to avoid problems with predictive models that only take numerical features
# dataframe -   dataframe to divide
# pred_col -    target variable
# val_replacer_origin - target values' transformation from categorical to numerical
# replacer -    new value for target value
def data_prep(dataframe, filename, pred_col, val_replacer_origin = None, replacer = None):
    features = dataframe.drop(columns = pred_col)
    if val_replacer_origin is not None and replacer is not None:
        target = dataframe[pred_col].replace(to_replace = val_replacer_origin, value = replacer)
    else:
        target = dataframe[pred_col]

    # Data profiling
    analysis(dataframe, filename, pred_col, replacer)

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 0)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test
