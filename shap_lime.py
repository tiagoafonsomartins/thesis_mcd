import pandas as pd
import numpy as np
import shap
import lime
import sklearn
import shap
import xgboost
import os
import auxiliary_functions as af
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

shap.initjs()  # load JS visualization code to notebook


def init_explainers(model, x_train, x_test, cols, target_values, dataset_name):
    lime_explainer(model, x_train, x_test, cols, target_values, dataset_name)
    shap_explainer(model, np.concatenate((x_train, x_test)), dataset_name)


# LIME explanation framework, takes four parameters as np.array and the black-box model
def lime_explainer(model, x_train, x_test, feature_labels, target_label, dataset_name, ID=None):
    # if isinstance(x_train, np.float64)
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=feature_labels, class_names=target_label,
                                                       discretize_continuous=False)
    if ID is None:
        np.random.randint(0, x_test.shape[0])

    exp = explainer.explain_instance(x_test[3], model.predict_proba, num_features=len(feature_labels))
    # exp.show_in_notebook(show_table=True, show_all=False)
    exp.save_to_file("results/explanations/"+dataset_name + "_lime_explanation.html")


# SHAP explanation framework, takes four parameters as np.array and the black-box model
def shap_explainer(model, x, dataset_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    shap_interactions = explainer.shap_interaction_values(x)

    shap.force_plot(explainer.expected_value, shap_values[0, :], x[0, :])

    shap.summary_plot(shap_values, x, plot_type="bar")
    shap.summary_plot(shap_interactions, x)
