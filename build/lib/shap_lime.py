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
shap.initjs() # load JS visualization code to notebook


# LIME explanation framework, takes four parameters as np.array and the black-box model
def lime_explainer(model, x_train, x_test, feature_labels, target_label, ID = None):
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names = feature_labels, class_names = target_label, discretize_continuous = False)
    if ID is None:
        np.random.randint(0, x_test.shape[0])

    exp = explainer.explain_instance(x_test.iloc[3], model.predict_proba, num_features = len(feature_labels))
    exp.show_in_notebook(show_table=True, show_all=False)


# SHAP explanation framework, takes four parameters as np.array and the black-box model
def shap_explainer(model, x):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    shap_interactions = explainer.shap_interaction_values(x)

    shap.force_plot(explainer.expected_value, shap_values, x)

    shap.summary_plot(shap_values, x, plot_type = "bar")
    shap.summary_plot(shap_interactions, x)

