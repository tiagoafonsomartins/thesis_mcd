import io

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

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


def save_to_file_2(filename, data_info, pred_col=None, pred_counts=None,
                   desc3="\n---Prediction column values---\n"):
    with open(filename, 'w') as file:
        for x in data_info:
            file.write(str(x))

        if pred_col is not None:
            file.write(desc3)
            file.write(str(pred_counts))


# Statistical analysis of obtained datasets
# dataframe -    Pandas DataFrame to be used for the analysis
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
# dataframe -   Pandas DataFrame to divide
# pred_col -    str with target variable
# val_replacer_origin - array with target values' transformation from categorical to numerical
# replacer -    array with new value for target value
def data_prep_sep_target(dataframe, filename, pred_col, seed, is_iapmei, val_replacer_origin=None, replacer=None):
    #seed=0
    features = dataframe.drop(columns=pred_col)
    if val_replacer_origin is not None and replacer is not None:
        target = dataframe[pred_col].replace(to_replace=val_replacer_origin, value=replacer)
    else:
        target = dataframe[pred_col]


    if is_iapmei:
        mic = dataframe.copy(deep=True)
        mic = mic[mic["Is micro enterprise"] == 1]
        mic_y = mic[pred_col]
        mic_x = mic.drop(columns=pred_col)
        x_train_mic, x_test_mic, y_train_mic, y_test_mic = train_test_split(mic_x, mic_y, test_size=0.3, random_state=seed, stratify=mic_y)

        peq = dataframe.copy(deep=True)
        peq = peq[peq["Is small company"] == 1]
        peq_y = peq[pred_col]
        peq_x = peq.drop(columns=pred_col)
        x_train_peq, x_test_peq, y_train_peq, y_test_peq = train_test_split(peq_x, peq_y, test_size=0.3, random_state=seed, stratify=peq_y)

        med = dataframe.copy(deep=True)
        med = med[med["Is medium company"] == 1]
        med_y = med[pred_col]
        med_x = med.drop(columns=pred_col)
        x_train_med, x_test_med, y_train_med, y_test_med = train_test_split(med_x, med_y, test_size=0.3, random_state=seed, stratify=med_y)

        x_train = np.concatenate((np.array(x_train_mic), np.array(x_train_peq), np.array(x_train_med)), axis=0)
        x_test = np.concatenate((np.array(x_test_mic), np.array(x_test_peq), np.array(x_test_med)), axis=0)
        y_train = np.concatenate((np.array(y_train_mic), np.array(y_train_peq), np.array(y_train_med)), axis=0)
        y_test = np.concatenate((np.array(y_test_mic), np.array(y_test_peq), np.array(y_test_med)), axis=0)
    else:
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0, stratify=target)
    # Data profiling
    #analysis(dataframe, filename, pred_col, replacer)

    ##train test split antigo era 80%-20% (nas exps iniciais 1 a 3) - usar se voltar a repetir exps de baseline
    #x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0, stratify=target)
    #x_train = np.array(x_train)
    #x_test = np.array(x_test)
    #y_train = np.array(y_train)
    #y_test = np.array(y_test)



    return x_train, x_test, y_train, y_test


# Alternative method to return only test and train data without separating the target feature
def data_prep(dataframe):
    train, test = train_test_split(dataframe, test_size=0.2, random_state=0)
    train = np.array(train)
    test = np.array(test)

    return train, test

def model_metrics(y, y_pred, title, filename, metrics):
    scores = pd.DataFrame()
    acc = accuracy_score(y, y_pred)
    #    roc_auc = roc_auc_score(target, model.predict_proba(pred), multi_class="ovr")
    # else:
    f1 = f1_score(y, y_pred)
    f1_weighted = f1_score(y, y_pred, average = "weighted")
    f1_macro = f1_score(y, y_pred, average = "macro")

    prec = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    # prec = precision_score(target, pred)
    # recall = recall_score(target, pred)
    # scores[title] = [acc, roc_auc, f1, prec, recall]
    scores[title] = [roc_auc, acc, f1, prec, recall]
    # scores.index = ['Accuracy', 'ROC_AUC', 'F1_Score', 'Precision_Score', 'Recall_Score']
    scores.index = ["ROC_AUC",'Accuracy', 'F1_Score', 'Precision_Score', 'Recall_Score']
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    scores["true negative"] = tn
    scores["false positive"] = fp
    scores["false negative"] = fn
    scores["true positive"] = tp
    metrics["ROC_AUC-"+title] = roc_auc
    metrics["Accuracy-"+title] = acc
    metrics["F1_Score-"+title] = f1
    metrics["F1_Score-weighted-"+title] = f1_weighted
    metrics["F1_Score-macro-"+title] =f1_macro
    metrics["Precision_Score-"+title] = prec
    metrics["Recall_Score-"+title] = recall
    if title == "Test":
        metrics["tn"] = tn
        metrics["fp"] = fp
        metrics["fn"] = fn
        metrics["tp"] = tp
    #save_to_file("results/model_performance/" + filename, scores.to_string(), "")
    #return scores
    return metrics

# Evaluate model performance
# model -
# title -
# feature -
# target -
# filename - str containing name of saved file
def model_evaluation(model, x_train, y_train, x_test, y_test, target_values, filename, metrics):

    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    model_metrics(y_train, y_pred, "Train", filename+"_train.txt", metrics)

    y_pred_2 = model.predict(x_test)
    model_metrics(y_test, y_pred_2, "Test", filename+"_test.txt", metrics)
    return metrics


def cat_names(data, categorical_features):
    categorical_names = {}
    data.columns = data.columns.str.strip()

    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[feature])
        data[feature] = le.transform(data[feature])
        categorical_names[feature] = le.classes_

    print(categorical_names)
    return categorical_names
