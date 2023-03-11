import pandas as pd
import sklearn

import auxiliary_functions as af
import xgboost

import explainers as exp


def analysis_explanation(dataset, model, dataset_cols_index, dataset_name, model_name, target_name, target_idx,
                         original_target_values, replacer, cat_cols, cat_cols_index, cont_cols):
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
    dataset.loc[-1] = dataset.columns
    dataset.index = dataset.index + 1
    dataset.sort_index(inplace=True)
    if original_target_values is not None and replacer is not None:
        dataset[target_name] = dataset[target_name].replace(to_replace=original_target_values, value=replacer)
    else:
        dataset[target_name] = dataset[target_name]
    dataset = dataset.astype(str)
    exp.permuteattack_explainer(model, dataset_no_target.columns, x_train, x_test, dataset_name)


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
default_credit_cat_cols_index = [1, 2, 3]
default_credit_cat_cols = ["Gender", "Education", "Marital status"]
default_credit_num_cols_num = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
default_credit_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
german_credit_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
german_credit_cat_cols_index = [ 2, 3, 5, 6, 8, 9]#, 11, 13, 14, 16, 18, 19]
iris_index = [0, 1, 2, 3]
heloc_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
default_credit_columns = ["Given credit (NT$)",
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
german_cat_cols = ["Status of existing checking account", "Credit history", "Purpose", "Savings account or bonds",
                   "Present employment since", "Personal status and sex", "Other debtors or guarantors",
                   "Property", "Other installment plans", "Housing", "Job", "Telephone", "Foreign worker"]  # , "risk"]
german_cont_cols = ["Duration in month", "Credit amount", "Install. rate (%) of disposable income",
                    "Present residence since", "Age in years", "No. of existing credits at this bank",
                    "No. people being liable for"]
german_cat_cols_no_target = german_cat_cols#.remove("risk")
german_cat_cols_num = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19, 20, 21]
german_cat_cols_no_target_num = german_cat_cols_num.remove(21)

german_num_cols_num = [1, 4, 7, 10, 12, 15, 17]
german_num_cols = ["duration", "amount", "installment_rate", "residence_since", "age", "credits_bank",
                   "liable_to_maintenance"]

# Reading the 3 datasets
# Default Credit target feat. values: 1 = Default; 0 = Not default
default_credit = pd.read_csv("datasets/default of credit card clients.csv", delimiter=';', header=0)
default_credit.columns = default_credit_columns
default_credit_anchors = pd.read_csv("datasets/default of credit card clients.csv", index_col=None, delimiter=';',
                                     header=None)
default_credit_anchors.columns = default_credit_columns
# german_credit = pd.read_csv("datasets/german_data.csv", delimiter=';', header=0)
iris = pd.read_csv("datasets/iris.csv", delimiter=';', header=None)
iris.columns = iris_target
german_credit_num = pd.read_csv("datasets/german.data-numeric.csv", delimiter=';', header=0)
german_credit_num = german_credit_num.drop(["cost_matrix_1", "cost_matrix_2", "cost_matrix_3", "cost_matrix_4"], axis=1)
german_credit_num.columns = german_credit_columns
heloc = pd.read_csv("datasets/heloc_dataset_v1.csv", delimiter=',', header=0)

# INICIO TESTE DATASET DEFAULT CREDIT
# Default Credit - Data preparation and split into train/test
# Black-box model
xgb_final = xgboost.XGBClassifier(tree_method='hist',
                                  n_estimators=800,
                                  min_child_weight=6,
                                  max_depth=2,
                                  gamma=0,
                                  eta=0.4,
                                  early_stop=10,
                                  random_state=42)

random_forest_classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)

analysis_explanation(default_credit, xgb_final, default_credit_index, "default_credit", "xgboost", "Y", 23, None,
                     [0, 1], default_credit_cat_cols, default_credit_cat_cols_index, default_credit_cont_cols)
# analysis_explanation(default_credit, random_forest_classifier, default_credit_index, "default_credit", "random_forest", "Y", 23, None,
#                     [0, 1], default_credit_cat_cols)
# FIM TESTE DATASET 1

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
analysis_explanation(iris, xgb_final, iris_index, "iris", "xgboost", "class", 4,
                     ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2], None, None, iris_no_target)
# analysis_explanation(iris, random_forest_classifier, iris_index, "iris", "xgboost", "class", 4,
#                     ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2], None)
# None, [0, 1, 2], None)
# FIM TESTE DATASET 1

# German Credit - Data preparation and split into train/test
xgb_final = xgboost.XGBClassifier(tree_method='hist',
                                  n_estimators=800,
                                  min_child_weight=6,
                                  max_depth=2,
                                  gamma=0,
                                  eta=0.4,
                                  early_stop=10,
                                  random_state=42)

random_forest_classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
analysis_explanation(german_credit_num, xgb_final, german_credit_index, "german_credit", "xgboost", "Risk", 20,
                     ['1', '2'], [0, 1], german_cat_cols, german_credit_cat_cols_index, german_cont_cols)
# analysis_explanation(german_credit_num, random_forest_classifier, german_credit_index, "german_credit", "xgboost", "Risk", 20,
#                     ['1', '2'], [0, 1], None)

# HELOC - Data preparation and split into train/test
xgb_final = xgboost.XGBClassifier(tree_method='hist',
                                  n_estimators=800,
                                  min_child_weight=6,
                                  max_depth=2,
                                  gamma=0,
                                  eta=0.4,
                                  early_stop=10,
                                  random_state=42)

random_forest_classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
#analysis_explanation(heloc, xgb_final, heloc_index, "heloc", "xgboost", "RiskPerformance", 21,
#                     ["Bad", "Good"], [0, 1], None)
