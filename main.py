import matplotlib
import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import auxiliary_functions as af
import xgboost

import explainers as exp


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

def model_constructor():

    #logistic_regressor = sklearn.linear_model.LogisticRegression(random_state=0)
    #svm_regressor = sklearn.svm.SVC(probability=True)
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

    return [ gaussian_naive_bayes, random_forest_classifier, mlp_regressor, xgb_final, decision_tree], ["gnb", "rf", "mlp", "xgb", "dt"]


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
#german_credit_num = pd.read_csv("datasets/german.data-numeric.csv", delimiter=';', header=0)
#german_credit_num = german_credit_num.drop(["cost_matrix_1", "cost_matrix_2", "cost_matrix_3", "cost_matrix_4"], axis=1)
#german_credit_num.columns = german_credit_columns
#heloc = pd.read_csv("datasets/heloc_dataset_v1.csv", delimiter=',', header=0)

#models, names = model_constructor()
#i = 0
#for model in models:
#    analysis_explanation(default_credit_initial, model, default_credit_index_sc, "default_credit_initial", names[i], "Y", 23, None,
#                         [0, 1], default_credit_cat_cols, default_credit_cat_index, default_credit_cont_cols)
#    print("model ", names[i])
#    i+=1

# INICIO TESTE DATASET DEFAULT CREDIT
# Default Credit - Data preparation and split into train/test
# Black-box model
#xgb_final = xgboost.XGBClassifier(tree_method='hist',
#                                  n_estimators=800,
#                                  min_child_weight=6,
#                                  max_depth=2,
#                                  gamma=0,
#                                  eta=0.4,
#                                  early_stop=10,
#                                  random_state=42)
#
#random_forest_classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
#analysis_explanation(default_credit_initial, xgb_final, default_credit_index_sc, "default_credit_initial", names[i], "Y", 23, None,
#                     [0, 1], default_credit_cat_cols, default_credit_cat_index, default_credit_cont_cols)


#models, names = model_constructor()
#i=0
#for model in models:
#    analysis_explanation(default_credit, model, default_credit_index_sc, "default_credit", names[i], "Y", 9, None,
#                         [0, 1], default_credit_cat_sc, default_credit_cat_index_sc, default_credit_cont_cols_sc)
#    print("model ", names[i])
#    i+=1

# Scaled Dataset
#analysis_explanation(default_credit, model, default_credit_index_sc, "default_credit", "xgboost", "Y", 9, None,
#                     [0, 1], default_credit_cat_sc, default_credit_cat_index_sc, default_credit_cont_cols_sc)

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

#models, names = model_constructor()
#i=0
#for model in models:
#    analysis_explanation(iris, xgb_final, iris_index, "iris", names[i], "class", 4,
#                         ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2], None, None, iris_no_target)
#    print("model ", names[i])
#    i+=1

#analysis_explanation(iris, xgb_final, iris_index, "iris", "xgboost", "class", 4,
#                     ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2], None, None, iris_no_target)
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

models, names = model_constructor()
i=0
for model in models:
    analysis_explanation(german_credit, model, german_credit_index, "german_credit", names[i], "Risk", 9,
                         ['1', '2'], [0, 1], german_cat_cols, german_credit_cat_cols_index, german_cont_cols)
    matplotlib.pyplot.close("all")
    print("model ", names[i])
    i+=1

#random_forest_classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
#analysis_explanation(german_credit, xgb_final, german_credit_index, "german_credit", "xgboost", "Risk", 20,
#                    ['1', '2'], [0, 1], german_cat_cols, german_credit_cat_cols_index, german_cont_cols)
# analysis_explanation(german_credit_num, random_forest_classifier, german_credit_index, "german_credit", "xgboost", "Risk", 20,
#                     ['1', '2'], [0, 1], None)


