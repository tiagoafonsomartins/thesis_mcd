import io

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


# Method to facilitate writing to a local file
# filename - name of file to be saved
# data_info - info to be saved
# data_desc - additional information
# pred_col - used for data analysis to determine target feature
# pred_counts - used for data analysis to determine counts of each possible value for target feature
# desc1 - description for data_info
# desc2 - description for data_desc
# desc3 - description for pred_col/pred_counts
def save_to_file(filename, data_info, data_desc="", pred_col=None, pred_counts=None,
                 desc1="---Data info---\n",
                 desc2="\n---Data Description---\n",
                 desc3="\n---Prediction column values---\n"):
    with open(filename, 'w') as file:
        file.write(desc1)
        file.write(data_info)
        file.write(desc2)
        file.write(data_desc)
        if pred_col is not None:
            file.write(desc3)
            file.write(str(pred_counts))


# Statistical analysis of obtained datasets
# dataframe -    DataFrame to be used for the analysis
# filename -     Name of output file. Should be similar to dataset used to avoid confusion
# pred_col -     Column used for predictions
# pred_values -  List of all possible values for the prediction
# noinspection PyDefaultArgument
def analysis(dataframe, filename="analysis.txt", pred_col=None, pred_values=[1, 0]):
    if isinstance(dataframe, pd.DataFrame):
        pred_counts = []

        if pred_col is not None:
            for pred_value in pred_values:
                print(pred_value)
                count = dataframe[pred_col].value_counts()[pred_value]
                pred_counts.append((pred_value, count))
                print("Number of predictions resulting in a value of ", pred_value, ": ", count)

        data_info = io.StringIO()
        dataframe.info(buf=data_info)
        data_info = data_info.getvalue()
        data_desc = dataframe.describe(include='all').to_string()
        print("Non-null values present in each column\n", data_info)
        print("Statistical information of each column\n", data_desc)
        save_to_file("results/analysis/" + filename, data_info, data_desc, pred_col, pred_counts,
                     "---Data info---\n",
                     "\n---Data Description---\n",
                     "\n---Prediction column values---\n")

    else:
        print("Given dataframe is not of type pandas.DataFrame")


# Prepare the data for prediction and explanation. It is suggested to use the last two parameters in order
# to avoid problems with predictive models that only take numerical features
# dataframe -   dataframe to divide
# pred_col -    target variable
# val_replacer_origin - target values' transformation from categorical to numerical
# replacer -    new value for target value
def data_prep_sep_target(dataframe, filename, pred_col, val_replacer_origin=None, replacer=None):
    features = dataframe.drop(columns=pred_col)
    if val_replacer_origin is not None and replacer is not None:
        target = dataframe[pred_col].replace(to_replace=val_replacer_origin, value=replacer)
    else:
        target = dataframe[pred_col]

    # Data profiling
    analysis(dataframe, filename, pred_col, replacer)

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test


# Alternative method to return only test and train data without separating the target feature
def data_prep(dataframe):

    train, test = train_test_split(dataframe, test_size=0.2, random_state=0)
    train = np.array(train)
    test = np.array(test)

    return train, test


def model_evaluation(model, title, feature, target, filename):
    scores = pd.DataFrame()
    pred = model.predict(feature)
    acc = accuracy_score(target, pred)
    roc_auc = roc_auc_score(target, pred)
    f1 = f1_score(target, pred)
    prec = precision_score(target, pred)
    recall = recall_score(target, pred)
    scores[title] = [acc, roc_auc, f1, prec, recall]
    scores.index = ['Accuracy', 'ROC_AUC', 'F1_Score', 'Precision_Score', 'Recall_Score']
    save_to_file("results/model_performance/" + filename, scores.to_string(), "")
    return scores


def cat_names(data, categorical_features):
    categorical_names = {}
    data.columns = data.columns.str.strip()

    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[feature])
        data[feature] = le.transform(data[feature])
        categorical_names[feature] = le.classes_
        #print("feature ", feature)
        #print("le ", le.fit(data.loc[:, feature]))
        #le.fit(data.loc[:, feature])
        #data.loc[:, feature] = le.transform(data.loc[:, feature])
        #categorical_names[feature] = le.classes_

    print(categorical_names)
    return categorical_names

