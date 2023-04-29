import matplotlib
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import auxiliary_functions as af
import xgboost
from datetime import datetime


import explainers as exp

def hyper_parm_optimization(dataset, target, dataset_name):
    y = dataset[target]
    X = dataset.drop(target, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    best_par = []
    logistic_regressor = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, random_state=0)
    param_grid = {"penalty": ["l2"],
                  "C": [10**-2, 10**-1, 10**0, 10**1, 10**2]}
    date_begin = datetime.now()
    gs = GridSearchCV(logistic_regressor, param_grid, cv=3, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)
    date_end = datetime.now()
    stat_info = gs.best_params_
    stat_info["date_begin"] = date_begin
    stat_info["date_end"] = date_end
    duration_s = (date_end - date_begin).total_seconds()
    stat_info["duration_s"] = duration_s.total_seconds()
    stat_info["duration_m"] = divmod(duration_s, 60)[0]
    stat_info["model_name"] = "Logistic Regression"
    best_par.append(stat_info)

    print("lr done")


    svm_regressor = sklearn.svm.SVR()
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
    gs = GridSearchCV(svm_regressor, param_grid, cv=3, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)
    date_end = datetime.now()
    stat_info = gs.best_params_
    stat_info["date_begin"] = date_begin
    stat_info["date_end"] = date_end
    duration_s = (date_end - date_begin).total_seconds()
    stat_info["duration_s"] = duration_s.total_seconds()
    stat_info["duration_m"] = divmod(duration_s, 60)[0]
    stat_info["model_name"] = "Support Vector Machine"
    best_par.append(stat_info)
    print("svm done")


    gaussian_naive_bayes = GaussianNB()
    param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
    gs = GridSearchCV(gaussian_naive_bayes, param_grid, cv=3, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)
    date_end = datetime.now()
    stat_info = gs.best_params_
    stat_info["date_begin"] = date_begin
    stat_info["date_end"] = date_end
    duration_s = (date_end - date_begin).total_seconds()
    stat_info["duration_s"] = duration_s.total_seconds()
    stat_info["duration_m"] = divmod(duration_s, 60)[0]
    stat_info["model_name"] = "Gaussian Naive-Bayes"

    best_par.append(stat_info)

    print("gnb done")
    random_forest_classifier = sklearn.ensemble.RandomForestClassifier(random_state=0)
    param_grid = {'bootstrap': [True, False],
                  'max_depth': [10, 20, 30, 40],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [200, 400, 600]}
    gs = GridSearchCV(random_forest_classifier, param_grid, cv=3, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)
    date_end = datetime.now()
    stat_info = gs.best_params_
    stat_info["date_begin"] = date_begin
    stat_info["date_end"] = date_end
    duration_s = (date_end - date_begin).total_seconds()
    stat_info["duration_s"] = duration_s.total_seconds()
    stat_info["duration_m"] = divmod(duration_s, 60)[0]
    stat_info["model_name"] = "Random Forest"

    best_par.append(stat_info)

    print("rf done")

    mlp_regressor = MLPClassifier()
    param_grid = parameter_space = {'hidden_layer_sizes': [(10, 30, 10), (20, 25)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive']}
    gs = GridSearchCV(mlp_regressor, param_grid, cv=3, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)
    date_end = datetime.now()
    stat_info = gs.best_params_
    stat_info["date_begin"] = date_begin
    stat_info["date_end"] = date_end
    duration_s = (date_end - date_begin).total_seconds()
    stat_info["duration_s"] = duration_s.total_seconds()
    stat_info["duration_m"] = divmod(duration_s, 60)[0]
    stat_info["model_name"] = "Multi-Layer Perceptron"

    best_par.append(stat_info)
    print("mlp done")


    xgb_final = xgboost.XGBClassifier(random_state=42)
    param_grid = {"colsample_bytree": [0.3, 0.5, 0.8],
        "reg_alpha": [0, 0.5, 1, 5],
        "reg_lambda": [0, 0.5, 1, 5]}
    gs = GridSearchCV(xgb_final, param_grid, cv=3, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)
    date_end = datetime.now()
    stat_info = gs.best_params_
    stat_info["date_begin"] = date_begin
    stat_info["date_end"] = date_end
    duration_s = (date_end - date_begin).total_seconds()
    stat_info["duration_s"] = duration_s.total_seconds()
    stat_info["duration_m"] = divmod(duration_s, 60)[0]
    stat_info["model_name"] = "XGBoost"

    best_par.append(stat_info)
    print("xgb done")


    decision_tree = sklearn.tree.DecisionTreeClassifier(random_state=0)
    param_grid = {'min_samples_leaf': [1, 2, 3],
        'max_depth': [1, 2, 3]}
    gs = GridSearchCV(decision_tree, param_grid, cv=3, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)
    date_end = datetime.now()
    stat_info = gs.best_params_
    stat_info["date_begin"] = date_begin
    stat_info["date_end"] = date_end
    duration_s = (date_end - date_begin).total_seconds()
    stat_info["duration_s"] = duration_s.total_seconds()
    stat_info["duration_m"] = divmod(duration_s, 60)[0]
    stat_info["model_name"] = "Decision Tree"
    best_par.append(stat_info)
    print("dt done")



    import json
    with open('results/best_parameters/'+dataset_name+"par", 'w') as fout:
        json.dump(best_par, fout)


def analysis_explanation(dataset, model, dataset_cols_index, dataset_name, model_name, target_name, target_idx,
                         original_target_values, replacer, cat_cols, cat_cols_index, cont_cols):
    pd.set_option('display.float_format', lambda x: '%0.2f' % x)
    x_train, x_test, y_train, y_test = af.data_prep_sep_target(dataset, dataset_name + ".txt", target_name,
                                                               val_replacer_origin=original_target_values,
                                                               replacer=replacer)

    if model_name != "xgboost":
        model.fit(x_train, y_train)

        #exp.anchor_explainer(model, dataset, target_idx, dataset.columns, dataset_cols_index,
        #                     cat_cols, dataset_name)
    else:
        model.fit(x_train, y_train)
    #exp.anchor_explainer(model, dataset, target_name, target_idx, dataset.columns, dataset_cols_index,
    #                     cat_cols_index, dataset_name)
    af.model_evaluation(model, "Train", x_train, y_train, len(replacer), dataset_name + "_" + model_name + "_train.txt")
    af.model_evaluation(model, "Test", x_test, y_test, len(replacer), dataset_name + "_" + model_name + "_test.txt")
     #dataset_no_target = dataset[:dataset.index(target_name)] + dataset[dataset.index(target_name)+1:]
    dataset_no_target = dataset.drop(str(target_name), axis=1)
    if len(replacer) > 2:
       multioutput = True
    else:
       multioutput = False
    exp.dice_explainer(dataset, model, cont_cols, target_name, replacer, dataset_name)
    exp.shap_explainer(model, x_train, dataset_no_target.columns, dataset_name, multioutput=multioutput)
    exp.lime_explainer(model, x_train, x_test, dataset.columns, replacer, dataset_name)
    for x in range(len(x_train[0])):
        exp.pdp_explainer(model, x_train, [x], dataset.columns, dataset_name, target_idx)
    dataset_permute = dataset.copy(deep=True)
    dataset_permute.loc[-1] = dataset_permute.columns
    dataset_permute.index = dataset_permute.index + 1
    dataset_permute.sort_index(inplace=True)
    if original_target_values is not None and replacer is not None:
        dataset_permute[target_name] = dataset_permute[target_name].replace(to_replace=original_target_values, value=replacer)
    else:
        dataset_permute[target_name] = dataset_permute[target_name]
    dataset_permute = dataset_permute.astype(str)
    #exp.permuteattack_explainer(model, dataset_no_target.columns, x_train, x_test, dataset_name)


# This function will initiate each predictive model with the best hyper-parameters
def model_constructor():

    logistic_regressor = sklearn.linear_model.LogisticRegression(random_state=0)
    svm_regressor = sklearn.svm.SVR(probability=True)
    gaussian_naive_bayes = GaussianNB()
    random_forest_classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
    mlp_regressor = MLPClassifier()
    xgb_final = xgboost.XGBClassifier(tree_method='hist',
                                      n_estimators=800,
                                      min_child_weight=6,
                                      max_depth=2,
                                      gamma=0,
                                      eta=0.4,
                                      early_stop=10,
                                      random_state=42)
    decision_tree = sklearn.tree.DecisionTreeClassifier()

    return [logistic_regressor, gaussian_naive_bayes, random_forest_classifier, mlp_regressor, xgb_final, decision_tree], ["logistic_regressor", "gnb", "rf", "mlp", "xgb", "dt"]


iris_target = ["sepal length (cm)",
               "sepal width (cm)",
               "petal length (cm)",
               "petal width (cm)",
               "class"]
iris_no_target = iris_target[:iris_target.index("class")] + iris_target[iris_target.index("class") + 1:]

# Defining categorical and numerical columns for Credit Card
# default_credit_cat_cols = ["X2", "X3", "X4"]
# REMOVE RETIRA DA LISTA ORIGINAL!!!!!
default_credit_num_cols = ["X1", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17",
                           "X18", "X19", "X20", "X21", "X22", "X23", "Y"]
default_credit_num_cols_no_target = default_credit_num_cols.remove("Y")
default_credit_cat_index = [1, 2, 3]
default_credit_cat_cols = ["Gender", "Education", "Marital status"]
default_credit_num_cols_num = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
default_credit_index_sc = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
german_credit_index = list(range(0, 47))
gc_cat_col_pt1 = [0, 1, 3, 4, 5, 6, 8]
gc_cat_col_pt2 = list(range(10, 47))
german_credit_cat_cols_index = gc_cat_col_pt1+gc_cat_col_pt2
iris_index = [0, 1, 2, 3]
heloc_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
default_credit_columns_initial = ["Given credit (NT$)",
                          "Gender",
                          "Education",
                          "Marital status",
                          "Age",
                          "Past, monthly payment (-1)",
                          "Past, monthly payment (-2)",
                          "Past, monthly payment (-3)",
                          "Past, monthly payment (-4)",
                          "Past, monthly payment (-5)",
                          "Past, monthly payment (-6)",
                          "Past, monthly bill (-1)",
                          "Past, monthly bill (-2)",
                          "Past, monthly bill (-3)",
                          "Past, monthly bill (-4)",
                          "Past, monthly bill (-5)",
                          "Past, monthly bill (-6)",
                          "Prev. payment in NT$ (-1)",
                          "Prev. payment in NT$ (-2)",
                          "Prev. payment in NT$ (-3)",
                          "Prev. payment in NT$ (-4)",
                          "Prev. payment in NT$ (-5)",
                          "Prev. payment in NT$ (-6)",
                          "Y"]


default_credit_cols = ["Given credit (NT$)",
"Education",
"Age",
"Past, monthly payment (-1)",
"Past, monthly payment (-2)",
"Past, monthly bill (-1)",
"Past, monthly bill (-2)",
"Prev. payment in NT$ (-1)",
"Prev. payment in NT$ (-2)",
"Y",
"Gender_female",
"Gender_male",
"Marital status_married",
"Marital status_others",
"Marital status_single"]



default_credit_index_sc = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

default_credit_cat_index_sc = [1, 3, 4, 10, 11, 12, 13, 14]

default_credit_cat_sc = ["Gender_female",
                      "Gender_male",
                      "Marital status_married",
                      "Marital status_others",
                      "Marital status_single",
                      "Past, monthly payment (-1)",
                      "Past, monthly payment (-2)"]

default_credit_cont_cols_sc = ["Given credit (NT$)",
                            "Age",
                            "Past, monthly bill (-1)",
                            "Past, monthly bill (-2)",
                            "Prev. payment in NT$ (-1)",
                            "Prev. payment in NT$ (-2)"]#[0, 2, 5, 6, 7, 8, 9]


default_credit_cont_cols = ["Given credit (NT$)",
                            "Age",
                            "Past, monthly payment (-1)",
                            "Past, monthly payment (-2)",
                            "Past, monthly payment (-3)",
                            "Past, monthly payment (-4)",
                            "Past, monthly payment (-5)",
                            "Past, monthly payment (-6)",
                            "Past, monthly bill (-1)",
                            "Past, monthly bill (-2)",
                            "Past, monthly bill (-3)",
                            "Past, monthly bill (-4)",
                            "Past, monthly bill (-5)",
                            "Past, monthly bill (-6)",
                            "Prev. payment in NT$ (-1)",
                            "Prev. payment in NT$ (-2)",
                            "Prev. payment in NT$ (-3)",
                            "Prev. payment in NT$ (-4)",
                            "Prev. payment in NT$ (-5)",
                            "Prev. payment in NT$ (-6)",
                            ]
german_credit_columns = ["Status of existing checking account",
                         "Duration in month",
                         "Credit history",
                         "Purpose",
                         "Credit amount",
                         "Savings account or bonds",
                         "Present employment since",
                         "Install. rate (%) of disposable income",
                         "Personal status and sex",
                         "Other debtors or guarantors",
                         "Present residence since",
                         "Property",
                         "Age in years",
                         "Other installment plans",
                         "Housing",
                         "No. of existing credits at this bank",
                         "Job",
                         "No. people being liable for",
                         "Telephone",
                         "Foreign worker",
                         "Risk"]
# Defining categorical and numerical columns for German Credit
german_cat_cols = ['Status of existing checking account',
                   'Duration in month',
                   'Savings account or bonds',
                   'Present employment since',
                   'Install. rate (%) of disposable income',
                   'Present residence since',
                   'No. of existing credits at this bank', 'Job',
                   'No. people being liable for',
                   'Risk',
                   'Credit history_all_paid_duly',
                   'Credit history_critical',
                   'Credit history_delay',
                   'Credit history_existing_duly_until_now',
                   'Credit history_none_paid_duly',
                   'Purpose_business',
                   'Purpose_car_new',
                   'Purpose_car_used',
                   'Purpose_domestic_appliances',
                   'Purpose_education',
                   'Purpose_furniture_equipment',
                   'Purpose_others',
                   'Purpose_radio_television',
                   'Purpose_repairs',
                   'Purpose_retraining',
                   'Personal status and sex_female_divorced_separated_married',
                   'Personal status and sex_male_divorced_separated',
                   'Personal status and sex_male_married_widowed',
                   'Personal status and sex_male_single',
                   'Other debtors or guarantors_coapplicant',
                   'Other debtors or guarantors_guarantor',
                   'Other debtors or guarantors_none',
                   'Property_car_other',
                   'Property_real estate',
                   'Property_soc_savings_life_insurance',
                   'Property_unknown',
                   'Other installment plans_bank',
                   'Other installment plans_none',
                   'Other installment plans_stores',
                   'Housing_free',
                   'Housing_own',
                   'Housing_rent',
                   'Telephone_none',
                   'Telephone_yes',
                   'Foreign worker_no',
                   'Foreign worker_yes']

german_cont_cols = ["Credit amount",
                    "Age in years"]
german_cat_cols_no_target = german_cat_cols#.remove("risk")
german_cat_cols_num = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19, 20, 21]
german_cat_cols_no_target_num = german_cat_cols_num.remove(21)
german_num_cols_num = [2, 7]
german_num_cols = ["duration", "amount", "installment_rate", "residence_since", "age", "credits_bank",
                   "liable_to_maintenance"]

# Reading the 3 datasets
# Default Credit target feat. values: 1 = Default; 0 = Not default
#default_credit = pd.read_csv("datasets/default of credit card clients.csv", delimiter=';', header=0)
default_credit = pd.read_csv("default_credit-vFINAL.csv", index_col=None, delimiter=';', header=0)
default_credit_initial = pd.read_csv("datasets/default of credit card clients.csv", delimiter=';', header=0)
default_credit_initial.columns = default_credit_columns_initial

#default_credit.columns = default_credit_columns
#default_credit_anchors = pd.read_csv("datasets/default of credit card clients.csv", index_col=None, delimiter=';',
#                                     header=None)
#default_credit_anchors.columns = default_credit_columns
german_credit = pd.read_csv("german_scaled.csv", delimiter=';', header=0)
iris = pd.read_csv("datasets/iris.csv", delimiter=';', header=None)
iris.columns = iris_target


# Define best hyper-parameters
hp_def_credit = hyper_parm_optimization(default_credit_initial, "Y", "default_credit_initial")
hp_def_credit_sc = hyper_parm_optimization(default_credit, "Y", "default_credit")
hp_iris = hyper_parm_optimization(iris, "class", "iris")
hp_german = hyper_parm_optimization(german_credit, "Risk", "german")


# INICIO TESTE DATASET DEFAULT CREDIT
models, names = model_constructor()
i = 0
for model in models:
    analysis_explanation(default_credit_initial, model, default_credit_index_sc, "default_credit_initial", names[i], "Y", 23, None,
                         [0, 1], default_credit_cat_cols, default_credit_cat_index, default_credit_cont_cols)
    #matplotlib.pyplot.close("all")
    print("model ", names[i])
    i+=1


# DEFAULT CREDIT - Scaled Dataset
models, names = model_constructor()
i=0
for model in models:
    analysis_explanation(default_credit, model, default_credit_index_sc, "default_credit", names[i], "Y", 9, None,
                         [0, 1], default_credit_cat_sc, default_credit_cat_index_sc, default_credit_cont_cols_sc)
    #matplotlib.pyplot.close("all")
    print("model ", names[i])
    i+=1


# INICIO TESTE DATASET IRIS
xgb_final = xgboost.XGBClassifier(tree_method='hist',
                                  n_estimators=800,
                                  min_child_weight=6,
                                  max_depth=2,
                                  gamma=0,
                                  eta=0.4,
                                  early_stop=10,
                                  random_state=42)

random_forest_classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)

models, names = model_constructor()
i=0
for model in models:
    analysis_explanation(iris, model, iris_index, "iris", names[i], "class", 4,
                         ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2], None, None, iris_no_target)
    #matplotlib.pyplot.close("all")
    print("model ", names[i])
    i+=1

# German Credit - Data preparation and split into train/test
models, names = model_constructor()
i=0
for model in models:
    analysis_explanation(german_credit, model, german_credit_index, "german_credit", names[i], "Risk", 9,
                         ['1', '2'], [0, 1], german_cat_cols, german_credit_cat_cols_index, german_cont_cols)
    #matplotlib.pyplot.close("all")
    print("model ", names[i])
    i+=1