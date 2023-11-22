import os
import sys

import matplotlib
import pandas as pd
import numpy as np
import sklearn
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids, NeighbourhoodCleaningRule, \
    EditedNearestNeighbours
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import auxiliary_functions as af
import xgboost
from datetime import datetime
from sklearn import tree

import explainers as exp


def hyper_parm_optimization(dataset, target, dataset_name, seed, metrics, sampler, is_iapmei, val=False):
    cv_amount = 3
    #dataset = dataset[dataset["not created through SMOTE"] == True]
    y = dataset[target]
    X = dataset.drop(target, axis=1)

    if is_iapmei:
        seed = 80
        mic = dataset.copy(deep=True)
        mic = mic[mic["Is micro enterprise"] == 1]
        mic_y = mic[target]
        mic_x = mic.drop(columns=target)
        x_train_mic, x_test_mic, y_train_mic, y_test_mic = train_test_split(mic_x, mic_y, test_size=0.3, random_state=seed, stratify=mic_y)
    #
        peq = dataset.copy(deep=True)
        peq = peq[peq["Is small company"] == 1]
        peq_y = peq[target]
        peq_x = peq.drop(columns=target)
        x_train_peq, x_test_peq, y_train_peq, y_test_peq = train_test_split(peq_x, peq_y, test_size=0.3, random_state=seed, stratify=peq_y)
    #
        med = dataset.copy(deep=True)
        med = med[med["Is medium company"] == 1]
        med_y = med[target]
        med_x = med.drop(columns=target)
        x_train_med, x_test_med, y_train_med, y_test_med = train_test_split(med_x, med_y, test_size=0.3, random_state=seed, stratify=med_y)
    #
        X_train = np.concatenate((np.array(x_train_mic), np.array(x_train_peq), np.array(x_train_med)), axis=0)
        y_train = np.concatenate((np.array(y_train_mic), np.array(y_train_peq), np.array(y_train_med)), axis=0)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.55, random_state=seed, stratify=y)


    #if considered project ID for the split in train/test data
    #data = dataset.copy(deep=True).sort_index()
    #data_train = data[0:int(data.shape[0]*0.7)]
    #data_test = data[int(data.shape[0]*0.7)+1:]
    #X_train = data_train.drop(target, axis=1)
    #y_train = data_train[target]

    #X_test = data_test.drop(target, axis=1)
    #y_test = data_test[target]

    scoring='neg_mean_absolute_error'
    #especificamente para exp. over
    resample = sampler
    if resample == "Under":
        res = RandomUnderSampler(random_state=seed)

    elif resample == "Over":
        res = RandomOverSampler(sampling_strategy='minority', random_state = seed)

    elif resample == 'Tomek':
        res = TomekLinks(sampling_strategy='majority', n_jobs = -1)#random_state=80, n_jobs = -1)

    elif resample == 'Smote':
        res = SMOTE(sampling_strategy='minority',random_state=seed, n_jobs = -1)

    elif resample == 'Adasyn':
        res = ADASYN(sampling_strategy="minority",random_state=seed, n_jobs = -1)

    elif resample == 'SmoteTomek':
        res = SMOTETomek(sampling_strategy='auto',random_state=seed, n_jobs = -1)

    elif resample == 'SmoteTeenn':
        res = SMOTEENN(sampling_strategy='minority',random_state=seed, n_jobs = -1)

    elif resample == 'Cluster':
        res = ClusterCentroids(sampling_strategy="auto",random_state=seed)

    elif resample == 'NeighbourhoodClean':
        res = NeighbourhoodCleaningRule(sampling_strategy="majority", n_jobs = -1)

    elif resample == 'NearestNeighbours':
        res = EditedNearestNeighbours(sampling_strategy="majority",n_jobs = -1)
    if resample != None:
        X_train, y_train = res.fit_resample(X_train, y_train)


    best_par = []
    logistic_regressor = sklearn.linear_model.LogisticRegression(solver='lbfgs',max_iter=1000, random_state=seed)
    param_grid = {'C': [0.1, 1, 10, 100, 500],
                  'penalty': ['l1', 'l2'],
                  'max_iter': [100,200,500,1000],
                  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga']}
    date_begin = datetime.now()
    gs = GridSearchCV(logistic_regressor, param_grid, cv=cv_amount, n_jobs=-1, scoring=scoring, refit=True).fit(X_train, y_train)
    date_end = datetime.now()
    stat_info = gs.best_params_
    stat_info["date_begin"] = date_begin
    stat_info["date_end"] = date_end
    duration_s = (date_end - date_begin).total_seconds()
    stat_info["duration_s"] = duration_s
    stat_info["duration_m"] = divmod(duration_s, 60)[0]
    stat_info["model_name"] = "Logistic Regression"
    best_par.append(stat_info)

    metrics["lr-cv"] = duration_s
    print("lr done")

    gaussian_naive_bayes = GaussianNB()
    date_begin = datetime.now()
    param_grid = {'var_smoothing': [1e-9, 1e-12, 1e-15, 1e-20, 1e-25] }
    gs = GridSearchCV(gaussian_naive_bayes, param_grid, cv=cv_amount, n_jobs=-1, scoring=scoring, refit=True, verbose=3).fit(X_train, y_train)
    date_end = datetime.now()
    stat_info = gs.best_params_
    stat_info["date_begin"] = date_begin
    stat_info["date_end"] = date_end
    duration_s = (date_end - date_begin).total_seconds()
    stat_info["duration_s"] = duration_s
    stat_info["duration_m"] = divmod(duration_s, 60)[0]
    stat_info["model_name"] = "Gaussian Naive-Bayes"

    best_par.append(stat_info)
    metrics["gnb-cv"] = duration_s

    print("gnb done")
    date_begin = datetime.now()

    random_forest_classifier = sklearn.ensemble.RandomForestClassifier(random_state=seed)
    param_grid = {'bootstrap': [True, False],
                  'n_estimators': [100, 200, 400],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'max_depth': [10, 20, 50, None],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'criterion' :['gini', 'entropy']} #depois remover , 200, 300, 500, 700
    gs = GridSearchCV(random_forest_classifier, param_grid, cv=cv_amount, n_jobs=-1, scoring=scoring, refit=True, verbose=3).fit(X_train, y_train)

    date_end = datetime.now()
    stat_info = gs.best_params_
    stat_info["date_begin"] = date_begin
    stat_info["date_end"] = date_end
    duration_s = (date_end - date_begin).total_seconds()
    stat_info["duration_s"] = duration_s
    stat_info["duration_m"] = divmod(duration_s, 60)[0]
    stat_info["model_name"] = "Random Forest"

    best_par.append(stat_info)
    metrics["rf-cv"] = duration_s

#

    print("rf done")

    mlp_regressor = MLPClassifier(early_stopping=True, validation_fraction=0.1)
    param_grid = {
        'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        'solver' : ['sgd', 'adam'], #não uso lbfgs porque falha sempre a convergir
        'hidden_layer_sizes': [
        (1,3,4),(1,2,3),
        (1,3,3), (50, 1)],
        'alpha': [0.001, 0.0000000001],
        "learning_rate_init": [0.005, 0.01, 0.2, 1],
        'learning_rate': ['constant', 'adaptive'],
        "max_iter": [200, 300]} #depois remover , 600, 800, 1000
#
    date_begin = datetime.now()
    gs = GridSearchCV(mlp_regressor, param_grid,  cv=cv_amount, n_jobs=-1, scoring=scoring, refit=True, verbose=3).fit(X_train, y_train)
    date_end = datetime.now()
    stat_info = gs.best_params_
    stat_info["date_begin"] = date_begin
    stat_info["date_end"] = date_end
    duration_s = (date_end - date_begin).total_seconds()
    stat_info["duration_s"] = duration_s
    stat_info["duration_m"] = divmod(duration_s, 60)[0]
    stat_info["model_name"] = "Multi-Layer Perceptron"

    best_par.append(stat_info)
    print("mlp done")
    metrics["mlp-cv"] = duration_s

    xgb_final = xgboost.XGBClassifier(random_state=seed)
    param_grid = {"booster": ["gbtree", "gblinear", "dart"],
                  'nthread': [-1],
                  'learning_rate': [0.1,0.01],
                  'max_depth': [5, 10,15, None],
                  'min_child_weight': [1, 3],
                  'subsample': [0.5, 0.7],
                  'colsample_bytree': [0.5, 0.7],
                  'n_estimators' : [100, 200,500],
                  'objective': ['reg:squarederror']
                  }
                  #"early_stopping_rounds": [10]}
    date_begin = datetime.now()
    gs = GridSearchCV(xgb_final, param_grid, cv=cv_amount, n_jobs=-1, scoring=scoring, verbose=3, refit=True).fit(X_train, y_train)
    date_end = datetime.now()
    stat_info = gs.best_params_
    stat_info["date_begin"] = date_begin
    stat_info["date_end"] = date_end
    duration_s = (date_end - date_begin).total_seconds()
    stat_info["duration_s"] = duration_s
    stat_info["duration_m"] = divmod(duration_s, 60)[0]
    stat_info["model_name"] = "XGBoost"

    best_par.append(stat_info)
    print("xgb done")
    metrics["xgb-cv"] = duration_s

    decision_tree = sklearn.tree.DecisionTreeClassifier(random_state=seed)
    param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
                  'max_depth' : [2,4,10,20, 7],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'criterion': ['gini','entropy']}
    date_begin = datetime.now()
    gs = GridSearchCV(decision_tree, param_grid, cv=cv_amount, n_jobs=-1, scoring=scoring, refit=True, verbose=3).fit(X_train, y_train)
    date_end = datetime.now()
    stat_info = gs.best_params_
    stat_info["date_begin"] = date_begin
    stat_info["date_end"] = date_end
    duration_s = (date_end - date_begin).total_seconds()
    stat_info["duration_s"] = duration_s
    stat_info["duration_m"] = divmod(duration_s, 60)[0]
    stat_info["model_name"] = "Decision Tree"
    best_par.append(stat_info)
    print("dt done")
    metrics["dt-cv"] = duration_s

    best_par.append({"cancel_train": str(len(list(filter(lambda y: y == 1, y_train)))), "closure_train": str(len(list(filter(lambda y: y == 0, y_train))))})

    if not val:
        import json
        with open('results/best_parameters/' + dataset_name + "par.txt", 'w') as fout:
            json.dump(best_par, fout, default=str)

    return best_par, metrics

def analysis_explanation(dataset, model, dataset_cols_index, dataset_name, model_name, target_name, target_idx,
                         original_target_values, replacer, cat_cols, cat_cols_index, cont_cols, metrics, seed, resampler, is_iapmei):
    pd.set_option('display.float_format', lambda x: '%0.2f' % x)
    x_train, x_test, y_train, y_test = af.data_prep_sep_target(dataset, dataset_name + ".txt", target_name, seed, is_iapmei,
                                                               val_replacer_origin=original_target_values,
                                                               replacer=replacer)
    data = dataset.copy(deep=True)
    x = data.drop(columns=target_name)
    y = data[target_name]
    with_explanations = True
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)

    #data = dataset.copy(deep=True).sort_index()
    #data_train = data[0:int(data.shape[0]*0.7)]
    #data_test = data[int(data.shape[0]*0.7)+1:]
    #x_train = data_train.drop(target_name, axis=1)
    #y_train = data_train[target_name]
#
    #x_test = data_test.drop(target_name, axis=1)
    #y_test = data_test[target_name]



    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)


    #especificamente para exp. SMOTE
    metrics["train_closure before sampler"] = len(list(filter(lambda y: y == 0, y_train)))
    metrics["train_cancelled before sampler"] = len(list(filter(lambda y: y == 1, y_train)))

    resample = resampler
    time_resample = datetime.now()
    if resample == "Under":
        res = RandomUnderSampler(random_state=seed)

    elif resample == "Over":
        res = RandomOverSampler(sampling_strategy='minority', random_state = seed)

    elif resample == 'Tomek':
        res = TomekLinks(sampling_strategy='majority', n_jobs = -1)#random_state=seed

    elif resample == 'Smote':
        res = SMOTE(sampling_strategy='minority',random_state=seed, n_jobs = -1)

    elif resample == 'Adasyn':
        res = ADASYN(sampling_strategy="minority",random_state=seed, n_jobs = -1)

    elif resample == 'SmoteTomek':
        res = SMOTETomek(sampling_strategy='auto',random_state=seed, n_jobs = -1)

    elif resample == 'SmoteTeenn':
        res = SMOTEENN(sampling_strategy='minority',random_state=seed, n_jobs = -1)

    elif resample == 'Cluster':
        res = ClusterCentroids(sampling_strategy="auto",random_state=seed)

    elif resample == 'NeighbourhoodClean':
        res = NeighbourhoodCleaningRule(sampling_strategy="majority", n_jobs = -1)

    elif resample == 'NearestNeighbours':
        res = EditedNearestNeighbours(sampling_strategy="majority",n_jobs = -1)

    if resample != None:
        x_train, y_train = res.fit_resample(x_train, y_train)

    time_resample_end = datetime.now()
    metrics["time to resample (s)"] = (time_resample_end - time_resample).total_seconds()

    metrics["train_closure after sampler"] = len(list(filter(lambda y: y == 0, y_train)))
    metrics["train_cancelled after sampler"] = len(list(filter(lambda y: y == 1, y_train)))
    metrics["test_closure"] = len(list(filter(lambda y: y == 0, y_test)))
    metrics["test_cancelled"] = len(list(filter(lambda y: y == 1, y_test)))

    if model_name == "dt":
        fig = plt.figure(figsize=(25, 20))
        _ = tree.plot_tree(model.fit(x_test, y_test),
                           feature_names=dataset.drop(target_name, axis=1).columns, class_names=True, proportion =True,
                           filled=True, max_depth=5)
        fig.savefig("results/explanations/"+dataset_name+"-decision_tree.svg")
        plt.clf()

    elif model_name == "xgboost":
        model.set_params({"early_stopping_rounds": [10]})
        model.fit(x_train, y_train)

        fig = plt.figure(figsize=(25, 20))
        _ = xgboost.plot_importance(model)
        fig.savefig("results/explanations/"+dataset_name+"-xgboost_feat_importance.svg")
        plt.clf()

        fig = plt.figure(figsize=(25, 20))
        _ = xgboost.plot_tree(model, num_trees=3)
        fig.savefig("results/explanations/"+dataset_name+"-xgboost_trees.svg")
        plt.clf()
        date_end = datetime.now()

    else:
        try:
            model.fit(x_train, y_train)
            date_end = datetime.now()
        except:
            print("failed to fit")


    #duration_s = (date_end - date_begin).total_seconds()
    #with open("results/number_feat_train_test/time/time_fit_train"+dataset_name+"-"+model_name+".txt", 'w') as file:
    #    file.write(str(duration_s))
    # exp.anchor_explainer(model, dataset, target_name, target_idx, dataset.columns, dataset_cols_index,
    #                     cat_cols_index, dataset_name)
    pred_start = datetime.now()
    try:
        metrics = af.model_evaluation(model, x_train, y_train, x_test, y_test, len(replacer), dataset_name + "_" + model_name, metrics)

    except:
        print("failed to evaluate")

    #af.model_evaluation(model, "Test", x_test, y_test, len(replacer), dataset_name + "_" + model_name + "_test.txt", metrics)
    pred_end = datetime.now()
    metrics["pred-s"] = (pred_end - pred_start).total_seconds()
    if with_explanations:
        #dataset_no_target = dataset[:dataset.index(target_name)] + dataset[dataset.index(target_name)+1:]
        dataset_no_target = dataset.drop(str(target_name), axis=1)
        if len(replacer) > 2:
            multioutput = True
        else:
            multioutput = False

        exp_start = datetime.now()
        exp.dice_explainer(dataset, pd.DataFrame(x_test, columns=dataset_no_target.columns), model, cont_cols, target_name, replacer, dataset_name, model_name)
        exp_end = datetime.now()
        metrics["dice-s"] = (exp_end -  exp_start).total_seconds()

        exp_start = datetime.now()
        exp.shap_explainer(model, x_train, dataset_no_target.columns, dataset_name, model_name, multioutput=multioutput)
        exp_end = datetime.now()
        metrics["shap-s"] = (exp_end -  exp_start).total_seconds()

        exp_start = datetime.now()
        exp.lime_explainer(model, x_train, x_test, dataset_no_target.columns, replacer, dataset_name, model_name)
        exp_end = datetime.now()
        metrics["lime-s"] = (exp_end -  exp_start).total_seconds()

        exp_start = datetime.now()
        for x in range(len(x_train[0])):
            exp.pdp_explainer(model, x_train, [x], dataset_no_target.columns, dataset_name, model_name) #target_idx
        exp_end = datetime.now()
        metrics["pdp-s"] = (exp_end -  exp_start).total_seconds()

        dataset_permute = dataset.copy(deep=True)
        dataset_permute.loc[-1] = dataset_permute.columns
        dataset_permute.index = dataset_permute.index + 1
        dataset_permute.sort_index(inplace=True)
        if original_target_values is not None and replacer is not None:
            dataset_permute[target_name] = dataset_permute[target_name].replace(to_replace=original_target_values,
                                                                                value=replacer)
        else:
            dataset_permute[target_name] = dataset_permute[target_name]
        dataset_permute = dataset_permute.astype(str)


        exp_start = datetime.now()
        exp.permuteattack_explainer(model, dataset_no_target.columns, x_train, x_test, dataset_name, model_name)
        exp_end = datetime.now()
        metrics["permuteattack-s"] = (exp_end -  exp_start).total_seconds()
        plt.close("all")

    else:
        metrics["dice-s"] = 0
        metrics["shap-s"] = 0
        metrics["lime-s"] = 0
        metrics["pdp-s"] = 0
        metrics["permuteattack-s"] = 0

    return metrics

# This function will initiate each predictive model with the best hyper-parameters given in arg parameters
#parameters - list of dict containing the best hyper-parameters for each model
def model_constructor(parameters):
    random_seed=80
    original_par_order = ["logistic_regressor", #"svc",
                           "gnb", "rf", "mlp", "xgb", "dt"]
    model_names = [ "dt","gnb", "lr",
                   #"svc",
                     "mlp","rf","xgb"]
    logistic_regressor = sklearn.linear_model.LogisticRegression(C=parameters[original_par_order.index("logistic_regressor")]["C"],
                                                                 penalty=parameters[original_par_order.index("logistic_regressor")]["penalty"],
                                                                 max_iter=parameters[original_par_order.index("logistic_regressor")]["max_iter"],
                                                                 solver=parameters[original_par_order.index("logistic_regressor")]["solver"],
                                                                 random_state=random_seed)

    gaussian_naive_bayes = GaussianNB(var_smoothing=parameters[original_par_order.index("gnb")]["var_smoothing"])

    random_forest_classifier = sklearn.ensemble.RandomForestClassifier(bootstrap=parameters[original_par_order.index("rf")]["bootstrap"],
                                                                       max_features=parameters[original_par_order.index("rf")]["max_features"],
                                                                       max_depth=parameters[original_par_order.index("rf")]["max_depth"],
                                                                       min_samples_leaf=parameters[original_par_order.index("rf")]["min_samples_leaf"],
                                                                       min_samples_split=parameters[original_par_order.index("rf")]["min_samples_split"],
                                                                       #min_weight_fraction_leaf=parameters[original_par_order.index("rf")]["min_weight_fraction_leaf"],
                                                                       n_estimators=parameters[original_par_order.index("rf")]["n_estimators"],
                                                                       criterion=parameters[original_par_order.index("rf")]["criterion"],
                                                                       random_state=random_seed)

    mlp_regressor = MLPClassifier(activation=parameters[original_par_order.index("mlp")]["activation"],
                                  alpha=parameters[original_par_order.index("mlp")]["alpha"],
                                  hidden_layer_sizes=parameters[original_par_order.index("mlp")]["hidden_layer_sizes"],
                                  learning_rate=parameters[original_par_order.index("mlp")]["learning_rate"],
                                  solver=parameters[original_par_order.index("mlp")]["solver"],
                                  learning_rate_init=parameters[original_par_order.index("mlp")]["learning_rate_init"],
                                  max_iter=parameters[original_par_order.index("mlp")]["max_iter"],
                                  random_state=random_seed)

    xgb_final = xgboost.XGBClassifier(colsample_bytree=parameters[original_par_order.index("xgb")]["colsample_bytree"],
                                      #reg_alpha=parameters[original_par_order.index("xgb")]["reg_alpha"],
                                      #reg_lambda=parameters[original_par_order.index("xgb")]["reg_lambda"],
                                      n_estimators=parameters[original_par_order.index("xgb")]["n_estimators"],
                                      min_child_weight=parameters[original_par_order.index("xgb")]["min_child_weight"],
                                      max_depth=parameters[original_par_order.index("xgb")]["max_depth"],
                                      #gamma=parameters[original_par_order.index("xgb")]["gamma"],
                                      learning_rate=parameters[original_par_order.index("xgb")]["learning_rate"],
                                      booster=parameters[original_par_order.index("xgb")]["booster"],
                                      objective=parameters[original_par_order.index("xgb")]["objective"]
                                      )

    decision_tree = sklearn.tree.DecisionTreeClassifier(max_depth=parameters[original_par_order.index("dt")]["max_depth"],
                                                        min_samples_leaf=parameters[original_par_order.index("dt")]["min_samples_leaf"],
                                                        max_features=parameters[original_par_order.index("dt")]["max_features"],
                                                        min_samples_split=parameters[original_par_order.index("dt")]["min_samples_split"],
                                                        criterion=parameters[original_par_order.index("dt")]["criterion"])

    return [decision_tree, gaussian_naive_bayes,
            #svm_classifier,
            logistic_regressor , mlp_regressor,random_forest_classifier
            , xgb_final], model_names

# Defining categorical and numerical columns for Credit Card
default_credit_num_cols = ["X1", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17",
                           "X18", "X19", "X20", "X21", "X22", "X23", "Y"]
default_credit_num_cols_no_target = default_credit_num_cols.remove("Y")
default_credit_cat_index = [1, 2, 3]
default_credit_cat_cols = ["Gender", "Education", "Marital status"]
default_credit_num_cols_num = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
german_credit_index = list(range(0, 47))
gc_cat_col_pt1 = [0, 1, 3, 4, 5, 6, 8]
gc_cat_col_pt2 = list(range(10, 47))
german_credit_cat_cols_index = gc_cat_col_pt1 + gc_cat_col_pt2
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
                               "Prev. payment in NT$ (-2)"]  # [0, 2, 5, 6, 7, 8, 9]

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
german_cat_cols_no_target = german_cat_cols  # .remove("risk")
german_cat_cols_num = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19, 20, 21]
german_cat_cols_no_target_num = german_cat_cols_num.remove(21)
german_num_cols_num = [2, 7]
german_num_cols = ["duration", "amount", "installment_rate", "residence_since", "age", "credits_bank",
                   "liable_to_maintenance"]

# Read the 2 public datasets
default_credit = pd.read_csv("datasets/default_credit_scaled.csv", index_col=None, delimiter=';', header=0)
default_credit_initial = pd.read_csv("datasets/default of credit card clients.csv", delimiter=';', header=0)
default_credit_initial.columns = default_credit_columns_initial
german_credit = pd.read_csv("datasets/german_scaled.csv", delimiter=';', header=0)


sampler_lst = [
    None,
    "Under",
    "Over",
    'Tomek',
    'Smote',
    'Adasyn',
    'SmoteTomek',
    'SmoteTeenn',
    'Cluster',
    'NeighbourhoodClean',
    'NearestNeighbours'
]

seeds = [
    0,
    351872,
    90415,
    727724,
    467374
]

#pipeline function. Starts with hyper-parameter search -> prediction -> explanation
def pipeline(is_iapmei, smote, dataset_name, dataset_folder, cols_idx, target, target_idx, original_target_values, replacer,
             cat_cols, cat_cols_idx, cont_cols):
    initial_seed = 0
    method_metrics = pd.DataFrame(columns=["sampler","seed","model","roc_auc","accuracy","f_score","precision",
                                           "recall","roc_auc","accuracy","f_score", "f_score - weighted", "f_score - macro", "precision","recall","TN","FP",
                                           "FN","TP","Closures on train before sampler","Cancellations on train before sampler",
                                           "Closures on train after sampler","Cancellations on train after sampler",
                                           "Closures on test","Cancellations on test",
                                           "time in gridsearch (s)","time to fit-predict (s)",
                                           "time to explain - DiCE (s)","time to explain - SHAP (s)",
                                           "time to explain - LIME (s)","time to explain - PDP (s)",
                                           "time to explain - PermuteAttack (s)",
                                           "time to resample (s)"])

    for sampler in sampler_lst:
        data = pd.read_csv(dataset_folder+dataset_name+".csv", header=0, encoding="utf-8", sep=";", index_col=None)
        grid_search_exec_start = datetime.now()
        hp, cv_metrics = hyper_parm_optimization(data, target, dataset_name + smote, initial_seed,
                                                        {}, sampler, is_iapmei)
        grid_search_exec_end = datetime.now()
        grid_search_diff = (grid_search_exec_end - grid_search_exec_start).total_seconds()
        for seed in seeds:

            main_dir = "results/explanations/"+dataset_name+smote+"-"+str(sampler)+"-"+str(seed)
            if not os.path.exists(main_dir+"/"):
                os.makedirs(main_dir+"/")

            import json
            with open('results/best_parameters/' + dataset_name + smote + "par", 'w') as fout:
                json.dump(hp, fout, default=str)
            # iapmei
            models, names = model_constructor(hp)
            i = 0
            pred_exp_exec_start = datetime.now()

            for model in models:
                if names[i] == "gnb":
                    debug="true"
                metrics = {}
                metrics = analysis_explanation(data, model, cols_idx, dataset_name+smote+"-"+str(sampler)+"-"+str(seed), names[i],
                                               target, target_idx, original_target_values,
                                               replacer, cat_cols,
                                               cat_cols_idx, cont_cols, metrics, seed, sampler, is_iapmei)

                print("model ", names[i])
                method_metrics.loc[len(method_metrics)] = [str(sampler),seed,names[i],metrics["ROC_AUC-Train"],
                                                           metrics["Accuracy-Train"],metrics["F1_Score-Train"],
                                                           metrics["Precision_Score-Train"],metrics["Recall_Score-Train"],
                                                           metrics["ROC_AUC-Test"],metrics["Accuracy-Test"],
                                                           metrics["F1_Score-Test"], metrics["F1_Score-weighted-Test"],
                                                           metrics["F1_Score-macro-Test"], metrics["Precision_Score-Test"],
                                                           metrics["Recall_Score-Test"],metrics["tn"],metrics["fp"],
                                                           metrics["fn"],metrics["tp"],
                                                           metrics["train_closure before sampler"],
                                                           metrics["train_cancelled before sampler"],
                                                           metrics["train_closure after sampler"],
                                                           metrics["train_cancelled after sampler"],
                                                           metrics["test_closure"],
                                                           metrics["test_cancelled"],
                                                           cv_metrics[names[i]+"-cv"],metrics["pred-s"],metrics["dice-s"],
                                                           metrics["shap-s"],metrics["lime-s"],metrics["pdp-s"],
                                                           metrics["permuteattack-s"], metrics["time to resample (s)"]
                                                           ]
                i += 1
            pred_exp_exec_end = datetime.now()
            pred_exp_exec_diff = (pred_exp_exec_end - pred_exp_exec_start).total_seconds()


        method_metrics.to_excel("results/model_performance/eval_bymodel-"+dataset_name+smote+"-"+str(sampler)+"-seed-"+str(seed)+".xlsx")


pipeline(False, "_post-dissertation", "datasets/german_scaled", "", german_credit_index, "Risk", 11,
         ['1', '2'], [0, 1], german_cat_cols, german_credit_cat_cols_index, german_cont_cols)


sys.exit(0)
pipeline(False, "_post-dissertation", "datasets/default_credit_scaled", "", default_credit_index_sc, "Y", 9, None,
         [0, 1], default_credit_cat_sc, default_credit_cat_index_sc, default_credit_cont_cols_sc)