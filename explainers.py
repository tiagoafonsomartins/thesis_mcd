import lime
import numpy as np
import pandas as pd
import shap
import matplotlib as plt
from sklearn.inspection import partial_dependence, PartialDependenceDisplay#, plot_partial_dependence
#from sklearn.inspection.partial_dependence import plot_partial_dependence
import auxiliary_functions as af
from models.PermuteAttack.src import ga_attack as permute
from models.anchor.anchor import utils as anchor_utils, anchor_tabular as anchor_tabular

# PermuteAttack initializer
# model - sklearn predictive model, tested after the operation fit() is made
# feature_names - array of strings, indicating the name of the columns of the dataset
# x_train/x_test - train/test dataset as numpy array
# idx - integer referring to the preferred instance for generation of explanation
def anchor_explainer(model, dataset, class_name, feature_names, features_to_use, categorical_features, dataset_name):

    dataset_anchors = anchor_utils.load_csv_dataset(dataset, features_to_use=features_to_use, feature_names=feature_names, categorical_features=None, target_idx=class_name, skip_first=True)


    explainer = anchor_tabular.AnchorTabularExplainer(
        dataset_anchors.class_names,
        dataset_anchors.feature_names,
        dataset_anchors.train,
        dataset_anchors.categorical_names)
    idx = 0
    np.random.seed(1)
    exp = explainer.explain_instance(np.array(dataset_anchors.test)[idx,:], model.predict, threshold=0.95)
    fit_anchor = np.where(np.all(np.array(dataset_anchors.test)[:, exp.features()] == np.array(dataset_anchors.test)[idx][exp.features()], axis=1))[0]
    fit_partial = np.where(np.all(np.array(dataset_anchors.test)[:, exp.features(1)] == np.array(dataset_anchors.test)[idx][exp.features(1)], axis=1))[0]
    anchors_data = []


    anchor_tmp ='Anchor: %s' % (' AND '.join(exp.names()))
    anchors_data.append(anchor_tmp + "\n")

    anchor_tmp ='Precision: %.2f' % exp.precision()
    anchors_data.append(anchor_tmp + "\n")

    anchor_tmp ='Coverage: %.2f' % exp.coverage()
    anchors_data.append(anchor_tmp + "\n")

    # Get test examples where the anchors pplies
    anchor_tmp = 'Anchor test precision: %.2f' % (np.mean(model.predict(np.array(dataset_anchors.test)[fit_anchor]) == model.predict(np.array(dataset_anchors.test)[idx].reshape(1, -1))))
    anchors_data.append(anchor_tmp + "\n")
    anchor_tmp = 'Anchor test coverage: %.2f' % (fit_anchor.shape[0] / float(np.array(dataset_anchors.test).shape[0]))
    anchors_data.append(anchor_tmp + "\n")
    anchor_tmp = 'Partial anchor: %s' % (' AND '.join(exp.names(1)))
    anchors_data.append(anchor_tmp + "\n")
    anchor_tmp = 'Partial precision: %.2f' % exp.precision(1)
    anchors_data.append(anchor_tmp + "\n")
    anchor_tmp = 'Partial coverage: %.2f' % exp.coverage(1)
    anchors_data.append(anchor_tmp + "\n")
    anchor_tmp = 'Partial anchor test precision: %.2f' % (np.mean(model.predict(np.array(dataset_anchors.test)[fit_partial]) == model.predict(np.array(dataset_anchors.test)[idx].reshape(1, -1))))
    anchors_data.append(anchor_tmp + "\n")
    anchor_tmp = 'Partial anchor test coverage: %.2f' % (fit_partial.shape[0] / float(np.array(dataset_anchors.test).shape[0]))
    af.save_to_file_2("results/explanations/"+dataset_name+"/anchors.txt", anchors_data)


def pdp_explainer(model, x_axis, features, feature_names, dataset_name, target=None):
    print("features \n", features, "\n", "feat names \n", feature_names, "\n")
    deciles = {0: np.linspace(0, 1, num=5)}

    if target != None:
        for target_class in model.classes_:
            print(target_class)
            pdp_results = partial_dependence(model, x_axis, features)
            pdp = PartialDependenceDisplay.from_estimator(model, x_axis, features=features, feature_names=feature_names, target=target_class)
            plt.pyplot.savefig("results/explanations/"+dataset_name+"/"+str(features[0])+"_target-"+str(target_class)+"_"+"_pdp_explanation.png")
    else:
        pdp_results = partial_dependence(model, x_axis, features)
        pdp = PartialDependenceDisplay.from_estimator(model, x_axis, features=features, feature_names=feature_names)
        plt.pyplot.savefig("results/explanations/"+dataset_name+"/"+str(features[0])+"_pdp_explanation.png")
#plot_partial_dependence(model, x_axis, features=features, feature_names=feature_names)

    #pdp = PartialDependenceDisplay([pdp_results], target_idx=0, features=features, feature_names=feature_names, deciles=deciles)
    #plt.pyplot.show()


# PermuteAttack initializer
# model - sklearn predictive model, tested after the operation fit() is made
# feature_names - array of strings, indicating the name of the columns of the dataset
# x_train/x_test - train/test dataset as numpy array
# idx - integer referring to the preferred instance for generation of explanation
def permuteattack_explainer(model, feature_names, x_train, x_test, dataset_name, idx=0):
    permute_attack = permute.GAdvExample(feature_names=list(feature_names),
                             sol_per_pop=30, num_parents_mating=15, cat_vars_ohe=None,
                             num_generations=200, n_runs=10, black_list=None,
                             verbose=False, beta=.95)

    x_all, x_changes, x_sucess = permute_attack.attack(model, x=x_test[idx, :], x_train=x_train)
    print("X_ALL \n", x_all, "\n", "X_CHANGES", "\n", x_changes, "\n", "X_SUCCESS", "\n", x_sucess)
    print("permute_attack.results ", permute_attack.results)
    #af.save_to_file_2("results/explanations/" + dataset_name + "/results_permuteattack_explanation.txt", pd.DataFrame(permute_attack.results).values)
    permute_tmp = []
    permute_tmp.append("x_all \n")
    permute_tmp.append(pd.DataFrame(x_all).values)
    permute_tmp.append("\n x_changes \n")
    permute_tmp.append(pd.DataFrame(x_changes).values)
    permute_tmp.append("\n x_success \n")
    permute_tmp.append(pd.DataFrame(x_sucess).values)
    #af.save_to_file_2("results/explanations/" + dataset_name + "/all_permuteattack_explanation.txt", pd.DataFrame(x_all).values)
    #af.save_to_file_2("results/explanations/" + dataset_name + "/changes_permuteattack_explanation.txt", pd.DataFrame(x_changes).values)
    #af.save_to_file_2("results/explanations/" + dataset_name + "/success_permuteattack_explanation.txt", pd.DataFrame(x_sucess).values)
    af.save_to_file_2("results/explanations/" + dataset_name + "/success_permuteattack_explanation.txt", permute_tmp)

'''
 Main method for SHAP/LIME explanations which aims to englobe the initialization of such methods, un-cluttering main.py
 model - sklearn predictive model, after .fit() method is applied to train data
 x_train/x_test - train/test data as numpy arrays
 cols - 
 target_values
 dataset_name - string representing the name of the dataset when saving the results
 '''
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
def shap_explainer(model, x, feature_names,  dataset_name, multioutput=False):
    shap.initjs()
    data = pd.DataFrame(x, columns=feature_names)
    #explainer = shap.KernelExplainer(model.predict_proba, data)

    #shap_values = explainer_tree.shap_values(data)
    if multioutput:
        explainer = shap.KernelExplainer(model.predict_proba, data)
        shap_values = explainer.shap_values(data)
        #fig = shap.waterfall(explainer_tree.expected_value[0], shap_values[0], x, show=False)
        shap.summary_plot(shap_values, data, class_names=model.classes_, show=False)
        plt.pyplot.savefig("results/explanations/"+dataset_name+"/shap_summary.png", bbox_inches="tight")
        plt.pyplot.clf()
        #shap.waterfall_plot(explainer.expected_value[0], shap_values[0][0], show=False)
        shap.force_plot(explainer.expected_value[0], shap_values[0], data)
        plt.pyplot.savefig("results/explanations/"+dataset_name+"/shap_force.png", bbox_inches="tight")
        plt.pyplot.clf()
    else:
        explainer = shap.Explainer(model)
        shap_values = explainer(data)
        shap.plots.waterfall(shap_values[0], show=False)
        plt.pyplot.savefig("results/explanations/"+dataset_name+"/shap_waterfall.png", bbox_inches="tight")
        plt.pyplot.clf()
        #fig = shap.force_plot(explainer_tree.expected_value, shap_values[0, :], x[0, :])
        #plt.pyplot.savefig("results/explanations/"+dataset_name+"/shap_force_plot.png")
        #plt.pyplot.clf()
        shap.summary_plot(shap_values, data, plot_type="bar", show=False)
        #fig = shap.summary_plot(shap_values.shap_values(data), data, plot_type="bar", show=False)
        plt.pyplot.savefig("results/explanations/"+dataset_name+"/shap_summary_plot.png", bbox_inches="tight")
        plt.pyplot.clf()

        #Feature interactions is not useful...
        #fig = shap.summary_plot(shap_interactions, data, show=False)
        for feat in feature_names:
            #fig = shap.plots.scatter(shap_values.shap_values(data)[:, feat], show=False)
            shap.plots.scatter(shap_values[:, feat], show=False)
            plt.pyplot.savefig("results/explanations/"+dataset_name+"/shap_scatter_"+feat+".png", bbox_inches="tight")
            plt.pyplot.clf()

        #shap.plots.force(explainer.expected_value, shap_values[0], data.iloc[0,:], show=False)
        #plt.pyplot.savefig("results/explanations/"+dataset_name+"/shap_force.png")
        #plt.pyplot.clf()
        shap.plots.bar(shap_values[0], show=False)
        plt.pyplot.savefig("results/explanations/"+dataset_name+"/shap_barplot.png", bbox_inches="tight")
        plt.pyplot.clf()





