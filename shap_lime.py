import numpy as np
import shap
import lime
import matplotlib as plt

shap.initjs()  # load JS visualization code to notebook

'''
 Main method for SHAP/LIME explanations which aims to englobe the initialization of such methods, un-cluttering main.py
 model - sklearn predictive model, after .fit() method is applied to train data
 x_train/x_test - train/test data as numpy arrays
 cols - 
 target_values
 dataset_name - string representing the name of the dataset when saving the results
 '''


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
    exp.save_to_file("results/explanations/" + dataset_name + "/lime_explanation.html")


# SHAP explanation framework, takes four parameters as np.array and the black-box model
def shap_explainer(model, x, dataset_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    shap_interactions = explainer.shap_interaction_values(x)

    shap.force_plot(explainer.expected_value, shap_values[0, :], x[0, :])

    fig = shap.summary_plot(shap_values, x, plot_type="bar", show=False)
    plt.pyplot.savefig("results/explanations/"+dataset_name+"/shap_explanation.png")
    fig = shap.summary_plot(shap_interactions, x, show=False)
    plt.pyplot.savefig("results/explanations/"+dataset_name+"/shap_explanation_2.png")