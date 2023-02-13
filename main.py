import pandas as pd
import sklearn

import auxiliary_functions as af
import xgboost

import explainers as exp

import importlib
#anchor = importlib.import_module("models.anchor-master.anchor")
#print(anchor)
#import anchor.utils
#import anchor.anchor_tabular
#from anchor.utils import map_array_values
#clear = importlib.import_module("models.CLEAR-master")
#
#dice = importlib.import_module("models.DiCE.dice_ml")
#
#anomaly = importlib.import_module("models.Locally-Interpretable-One-Class-Anomaly-Detection-for-Credit-Card-Fraud-Detection-main")
#
#pastle = importlib.import_module("models.PASTLE-main.src")
#
#permuteattack = importlib.import_module("PermuteAttack-main.src")


# Defining categorical and numerical columns for Credit Card
default_credit_cat_cols = ["X2", "X3", "X4"]
default_credit_num_cols = ["X1","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20","X21","X22","X23","Y"]
default_credit_num_cols_no_target = default_credit_num_cols.remove("Y")
default_credit_cat_cols_num = [2, 3, 4]
default_credit_num_cols_num = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
default_credit_index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22, 23]

#default_credit_no_target = ["X1","X2","X3","X4","X5","X6", "X7", "test", "test", "test", "X11", "X12", "X13", "X14", "X15", "X16", "X17","X18", "X19", "X20", "X21", "X22", "X23"]
#default_credit_with_target = ["X1","X2","X3","X4","X5","X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17","X18", "X19", "X20", "X21", "X22", "X23", "Y"]
default_credit_no_target = ["Given credit (NT$)",
                              "Gender",
                              "Education",
                              "Marital status",
                              "Age",
                              "Past, monthly payment (-6)",
                              "Past, monthly payment (-5)",
                              "Past, monthly payment (-4)",
                              "Past, monthly payment (-3)",
                              "Past, monthly payment (-2)",
                              "Past, monthly payment (-1)",
                              "Past, monthly bill (-6)",
                              "Past, monthly bill (-5)",
                              "Past, monthly bill (-4)",
                              "Past, monthly bill (-3)",
                              "Past, monthly bill (-2)",
                              "Past, monthly bill (-1)",
                              "Prev. payment in NT$ (-6)",
                              "Prev. payment in NT$ (-5)",
                              "Prev. payment in NT$ (-4)",
                              "Prev. payment in NT$ (-3)",
                              "Prev. payment in NT$ (-2)",
                              "Prev. payment in NT$ (-1)"]
default_credit_with_target = ["Given credit (NT$)",
                              "Gender",
                              "Education",
                              "Marital status",
                              "Age",
                              "Past, monthly payment (-6)",
                              "Past, monthly payment (-5)",
                              "Past, monthly payment (-4)",
                              "Past, monthly payment (-3)",
                              "Past, monthly payment (-2)",
                              "Past, monthly payment (-1)",
                              "Past, monthly bill (-6)",
                              "Past, monthly bill (-5)",
                              "Past, monthly bill (-4)",
                              "Past, monthly bill (-3)",
                              "Past, monthly bill (-2)",
                              "Past, monthly bill (-1)",
                              "Prev. payment in NT$ (-6)",
                              "Prev. payment in NT$ (-5)",
                              "Prev. payment in NT$ (-4)",
                              "Prev. payment in NT$ (-3)",
                              "Prev. payment in NT$ (-2)",
                              "Prev. payment in NT$ (-1)",
                              "Y"]

# Defining categorical and numerical columns for German Credit
german_cat_cols = ["status","history","purpose","savings","employment_since","personal_status","debtors", "property","installment_plans","housing","job","telephone","foreign_worker","risk"]
german_cat_cols_no_target = german_cat_cols.remove("risk")
german_cat_cols_num = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19, 20, 21]
german_cat_cols_no_target_num = german_cat_cols_num.remove(21)

german_num_cols_num = [1, 4, 7, 10, 12, 15, 17]
german_num_cols = ["duration","amount","installment_rate","residence_since","age","credits_bank","liable_to_maintenance"]


# Reading the 3 datasets
default_credit = pd.read_csv("datasets/default of credit card clients.csv", delimiter=';', header=0)
default_credit.columns = default_credit_with_target
default_credit_anchors = pd.read_csv("datasets/default of credit card clients.csv", index_col=None, delimiter=';', header=None)
default_credit_anchors.columns = default_credit_with_target
#german_credit = pd.read_csv("datasets/german_data.csv", delimiter=';', header=0)
iris = pd.read_csv("datasets/iris.csv", delimiter=',', header=0)
german_credit_num = pd.read_csv("datasets/german.data-numeric.csv", delimiter=';', header=0)
heloc = pd.read_csv("datasets/heloc_dataset_v1.csv", delimiter=',', header=0)

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

#INICIO TESTE DATASET 1
# Default Credit - Data preparation and split into train/test
x_train, x_test, y_train, y_test = af.data_prep_sep_target(default_credit, "default_credit.txt", "Y", replacer=[0, 1])
xgb_final.fit(x_train, y_train)
af.model_evaluation(xgb_final, "Train", x_train, y_train, "default_credit_xgboost_train.txt")
af.model_evaluation(xgb_final, "Test", x_test, y_test, "default_credit_xgboost_test.txt")

exp.shap_explainer(xgb_final, x_train, default_credit_no_target, "default_credit")
exp.lime_explainer(xgb_final, x_train, x_test, default_credit.columns, [0, 1], "default_credit")
#train, test = af.data_prep(default_credit)
#default_credit_anchors = anchor.utils.load_csv_dataset(default_credit)

random_forest_classifier.fit(x_train, y_train)
exp.anchor_explainer(random_forest_classifier, default_credit_anchors.values, 23, default_credit_anchors.columns, default_credit_index, default_credit_cat_cols_num, "default_credit")

exp.permuteattack_explainer(random_forest_classifier, default_credit_no_target, x_train, x_test, "default_credit")
for x in range(len(x_train[0])):
    #pdp = exp.pdp_explainer(random_forest_classifier, x_train, [x], [default_credit_with_target[x]], "default_credit")
    pdp = exp.pdp_explainer(random_forest_classifier, x_train, [x], default_credit_with_target, "default_credit")

#FIM TESTE DATASET 1
#pdp = exp.pdp_explainer(random_forest_classifier, x_train, [9], default_credit_no_target, "default_credit")

# German Credit - Data preparation and split into train/test
# x_train, x_test, y_train, y_test = af.data_prep(german_credit, "german_credit.txt", "risk", replacer = [1, 2])












# German Credit - Data preparation and split into train/test
x_train, x_test, y_train, y_test = af.data_prep_sep_target(german_credit_num, "german_credit.txt", "risk", ["1", "2"],
                                                           replacer=[0, 1])
xgb_final.fit(x_train, y_train)

af.model_evaluation(xgb_final, "Train", x_train, y_train, "german_credit_xgboost_train.txt")
af.model_evaluation(xgb_final, "Test", x_test, y_test, "german_credit_xgboost_test.txt")

exp.lime_explainer(xgb_final, x_train, x_test, german_credit_num.columns, [0, 1], "german_credit")








# HELOC - Data preparation and split into train/test
x_train, x_test, y_train, y_test = af.data_prep_sep_target(heloc, "heloc.txt", ["RiskPerformance"], ["Good", "Bad"],
                                                           replacer=[0, 1])
xgb_final.fit(x_train, y_train)
af.model_evaluation(xgb_final, "Train", x_train, y_train, "heloc_xgboost_train.txt")
af.model_evaluation(xgb_final, "Test", x_test, y_test, "heloc_xgboost_test.txt")

# sl.shap_explainer(xgb_final, np.concatenate((x_train, x_test)))
exp.lime_explainer(xgb_final, x_train, x_test, heloc.columns, [0, 1], "heloc")
#anchors.anchor_explanation(xgb_final, x_test+y_test, "RiskPerformance", heloc.columns, , {})
