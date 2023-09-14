import lime
import matplotlib.pyplot
import numpy as np
import pandas as pd
import shap
import matplotlib as plt
from sklearn.inspection import partial_dependence, PartialDependenceDisplay  # , plot_partial_dependence
# from sklearn.inspection.partial_dependence import plot_partial_dependence
import auxiliary_functions as af
from models.PermuteAttack.src import ga_attack as permute
from models.anchor.anchor import utils as anchor_utils, anchor_tabular as anchor_tabular
import models.DiCE.dice_ml
from models.DiCE.dice_ml.utils import helpers


# PermuteAttack initializer
# model - sklearn predictive model, tested after the operation fit() is made
# feature_names - array of strings, indicating the name of the columns of the dataset
# x_train/x_test - train/test dataset as numpy array
# idx - integer referring to the preferred instance for generation of explanation
def anchor_explainer(model, dataset, target_name, class_idx, feature_names, features_to_use, categorical_features, dataset_name):
    dataset_anchors = anchor_utils.load_csv_dataset(dataset.values, features_to_use=features_to_use,
                                                    feature_names=feature_names, categorical_features=None,
                                                    target_idx=class_idx, skip_first=True, discretize=True)
    dir = '/datasets/'
    #dataset_anchors = anchor_utils.load_dataset('adult', balance=True, dataset_folder=dir, discretize=True)
    explainer = anchor_tabular.AnchorTabularExplainer(
        dataset_anchors.class_names,
        dataset_anchors.feature_names,
        dataset_anchors.train,
        dataset_anchors.categorical_names)
    idx = 0
    np.random.seed(1)
    exp = explainer.explain_instance(np.array(dataset.drop(columns=target_name))[idx, :], model.predict, threshold=0.95)
    fit_anchor = np.where(
        np.all(np.array(dataset_anchors.test)[:, exp.features()] == np.array(dataset_anchors.test)[idx][exp.features()],
               axis=1))[0]
    fit_partial = np.where(np.all(
        np.array(dataset_anchors.test)[:, exp.features(1)] == np.array(dataset_anchors.test)[idx][exp.features(1)],
        axis=1))[0]
    anchors_data = []

    anchor_tmp = 'Anchor: %s' % (' AND '.join(exp.names()))
    anchors_data.append(anchor_tmp + "\n")

    anchor_tmp = 'Precision: %.2f' % exp.precision()
    anchors_data.append(anchor_tmp + "\n")

    anchor_tmp = 'Coverage: %.2f' % exp.coverage()
    anchors_data.append(anchor_tmp + "\n")

    # Get test examples where the anchors pplies
    anchor_tmp = 'Anchor test precision: %.2f' % (np.mean(
        model.predict(np.array(dataset_anchors.test)[fit_anchor]) == model.predict(
            np.array(dataset_anchors.test)[idx].reshape(1, -1))))
    anchors_data.append(anchor_tmp + "\n")
    anchor_tmp = 'Anchor test coverage: %.2f' % (fit_anchor.shape[0] / float(np.array(dataset_anchors.test).shape[0]))
    anchors_data.append(anchor_tmp + "\n")
    anchor_tmp = 'Partial anchor: %s' % (' AND '.join(exp.names(1)))
    anchors_data.append(anchor_tmp + "\n")
    anchor_tmp = 'Partial precision: %.2f' % exp.precision(1)
    anchors_data.append(anchor_tmp + "\n")
    anchor_tmp = 'Partial coverage: %.2f' % exp.coverage(1)
    anchors_data.append(anchor_tmp + "\n")
    anchor_tmp = 'Partial anchor test precision: %.2f' % (np.mean(
        model.predict(np.array(dataset_anchors.test)[fit_partial]) == model.predict(
            np.array(dataset_anchors.test)[idx].reshape(1, -1))))
    anchors_data.append(anchor_tmp + "\n")
    anchor_tmp = 'Partial anchor test coverage: %.2f' % (
                fit_partial.shape[0] / float(np.array(dataset_anchors.test).shape[0]))
    af.save_to_file_2("results/explanations/" + dataset_name + "/anchors.txt", anchors_data)


def pdp_explainer(model, x_axis, features, feature_names, dataset_name, model_name,  target=None):
    print("features \n", features, "\n", "feat names \n", feature_names, "\n")
    deciles = {0: np.linspace(0, 1, num=5)}
    #

    if target != None:
        if len(model.classes_) > 2:
            for target_class in model.classes_:
                print(target_class)
                pdp_results = partial_dependence(model, x_axis, features)
                pdp = PartialDependenceDisplay.from_estimator(model, x_axis, features=features, feature_names=feature_names,
                                                              target=target_class)
                plt.pyplot.savefig("results/explanations/" + dataset_name + "/" + str(features[0]) + "_target-" + str(
                    target_class) + "_" + "_pdp_explanation_" + model_name + ".png", dpi=1200)
        else:
            pdp_results = partial_dependence(model, x_axis, features)
            pdp = PartialDependenceDisplay.from_estimator(model, x_axis, features=features, feature_names=feature_names)
            plt.pyplot.savefig("results/explanations/" + dataset_name + "/" + str(features[0]) + "_" + "_pdp_explanation_" + model_name + ".png", dpi=1200)
    else:
        pdp_results = partial_dependence(model, x_axis, features)
        pdp = PartialDependenceDisplay.from_estimator(model, x_axis, features=features, feature_names=feature_names)
        plt.pyplot.savefig("results/explanations/" + dataset_name + "/" + str(features[0]) + "_pdp_explanation_" + model_name + ".png", dpi=1200)
    plt.pyplot.clf()
    plt.pyplot.close("all")
    #matplotlib.pyplot.close("all")


# plot_partial_dependence(model, x_axis, features=features, feature_names=feature_names)

# pdp = PartialDependenceDisplay([pdp_results], target_idx=0, features=features, feature_names=feature_names,
# deciles=deciles) plt.pyplot.show()


# PermuteAttack initializer
# model - sklearn predictive model, tested after the operation fit() is made
# feature_names - array of strings, indicating the name of the columns of the dataset
# x_train/x_test - train/test dataset as numpy array
# idx - integer referring to the preferred instance for generation of explanation
def permuteattack_explainer(model, feature_names, x_train, x_test, dataset_name, model_name,  idx=0):
    pd.set_option('display.float_format', lambda x: '%0.2f' % x)
    permute_attack = permute.GAdvExample(feature_names=list(feature_names),
                                         sol_per_pop=30, num_parents_mating=15, cat_vars_ohe=None,
                                         num_generations=200, n_runs=10, black_list=None,
                                         verbose=False, beta=.95)

    x_all, x_changes, x_sucess = permute_attack.attack(model, x=x_test[idx, :], x_train=x_train)
    print("X_ALL \n", x_all, "\n", "X_CHANGES", "\n", x_changes, "\n", "X_SUCCESS", "\n", x_sucess)
    print("permute_attack.results ", permute_attack.results)
    # af.save_to_file_2("results/explanations/" + dataset_name + "/results_permuteattack_explanation.txt",
    # pd.DataFrame(permute_attack.results).values)
    plt.pyplot.clf()
    plt.pyplot.close("all")
    #permute_temp saves data for a single prediction, with the counterfactual predictions
    permute_tmp = []
    permute_tmp.append("x_all \n")
    permute_tmp.append(pd.DataFrame(x_all).values)
    permute_tmp.append("\n x_changes \n")
    permute_tmp.append(pd.DataFrame(x_changes).values)
    permute_tmp.append("\n x_success \n")
    permute_tmp.append(pd.DataFrame(x_sucess).values)
    #permute_tmp.append("\n results \n")
    #pd.set_option('display.float_format', str)
    #permute_tmp.append(pd.DataFrame(permute_attack.results.data).round(3).values)
    permute_tmp.append("\n original instance \n")
    permute_tmp.append(pd.DataFrame(x_test[idx, :]).values)
    #to_save = pd.concat([pd.DataFrame(x_test[idx, :]).transpose(), pd.DataFrame(x_sucess)])
    if permute_attack.results is not None:
        to_save = permute_attack.results.data
    #to_save.columns = feature_names
        to_save.to_csv("results/explanations/" + dataset_name + "/permuteattack_explanation_" + model_name + ".csv", index=False)
    af.save_to_file_2("results/explanations/" + dataset_name + "/success_permuteattack_explanation_" + model_name + ".txt", permute_tmp)

    #visualization of data regarding the entire dataset
    results = []

    i = 0
    print("X_TEST", len(x_test))
    plt.pyplot.style.use('ggplot')

    for xi in x_test[0:60]:#int(len(x_test)/4)]:
        x_all, x_changes, x_sucess = permute_attack.attack(model, x=xi,x_train=x_train)
        print("X_TEST ITERATION", i)
        i+=1
        if len(x_sucess)>0:
            results.append((xi,x_changes, x_sucess))

    adv = []
    data = []
    for result in results:
        for xi in result[2]:
            adv.append(xi)
            data.append(result[0])


    if len(data) > 0 and len(adv) > 0:
        plt.pyplot.style.use('ggplot')

        predprob_data = model.predict_proba(data)[:,1]
        predprob_adv = model.predict_proba(adv)[:,1]
        pred = model.predict(data)

        # Histogram for original vs counterfactual predictions, for prediction class 1
        plt.pyplot.clf()
        plt.pyplot.close("all")
        plt.pyplot.hist(predprob_data[pred==1], label="Original Prediction")
        plt.pyplot.hist(predprob_adv[pred==1], label="Counterfactual Prediction")
        plt.pyplot.legend(loc='best', fontsize=13)
        plt.pyplot.gca().set_xlabel("Prediction Probability", fontsize=13)
        plt.pyplot.gca().set_ylabel("Counts", fontsize=13)
        plt.pyplot.savefig("results/explanations/" + dataset_name + "/permuteattack_explanation_histogram_"+ model_name + ".png", dpi=1200)

        plt.pyplot.clf()
        plt.pyplot.close("all")
        # Histogram for original vs counterfactual predictions, for prediction class 0
        plt.pyplot.hist(predprob_data[pred==0], label="Original Prediction")
        plt.pyplot.hist(predprob_adv[pred==0], label="Counterfactual Prediction")
        plt.pyplot.legend(loc='best', fontsize=13)
        plt.pyplot.gca().set_xlabel("Prediction Probability", fontsize=13)
        plt.pyplot.gca().set_ylabel("Counts", fontsize=13)

        #fig = plt.pyplot.figure(figsize=(7,7))
        plt.pyplot.savefig("results/explanations/" + dataset_name + "/permuteattack_explanation_histogram_2_" + model_name + ".png", dpi=1200)
        plt.pyplot.clf()
        plt.pyplot.close("all")
        # Distributions of original/counterfactual predictions

        plt.pyplot.scatter(predprob_data[pred==1], predprob_adv[pred==1], c="b", vmin=0, vmax=1, label = "Original Pred = 1, Counterfactual Pred = 0")
        plt.pyplot.scatter(predprob_data[pred==0], predprob_adv[pred==0], c="r", vmin=0, vmax=1, label = "Original Pred = 0, Counterfactual Pred = 1")

        plt.pyplot.hlines(0.5, 0, 1, linestyles="--")
        plt.pyplot.vlines(0.5, 0, 1, linestyles="--")

        plt.pyplot.legend(loc='best', fontsize=13)
        plt.pyplot.gca().set_xlabel("Original Probability", fontsize=15)
        plt.pyplot.gca().set_ylabel("Counterfactual Probability", fontsize=15)
        plt.pyplot.tight_layout()
        plt.pyplot.savefig("results/explanations/" + dataset_name + "/permuteattack_explanation_scatter" + model_name + ".png", dpi=1200)
        #plt.pyplot.savefig("results/explanations/" + dataset_name + "/permuteattack_explanation_scatter.png", type="png", dpi=600)
        plt.pyplot.clf()
        plt.pyplot.close("all")

        # Number of features changed in order to get desired output (opposite class)
        x_change_all = results[0][1]
        for result in results[1:]:
            x_change_all = pd.concat([x_change_all, result[1]])

        x_change_all.head()
        ind = model.predict(adv)==0
        df1 = (x_change_all[ind]).sum(0).to_frame().sort_values(by=0,ascending=False)

        # se alguma parte dos graficos estiver vazio e porque não foi possível criar exp contrafactuar
        if df1.shape[1] > 1:

            plt.pyplot.figure(figsize=(10,6))
            ax= plt.pyplot.subplot(1,2,1)

            ax = df1[df1[0]!=0].plot.bar(legend=False, fontsize=13, ax = ax)
            ax.set_ylabel("Counts", fontsize=14)
            ax.set_title("Pred. changed from Default to Not Default")

        ind = model.predict(adv)==1
        df2 = (x_change_all[ind]).sum(0).to_frame().sort_values(by=0,ascending=False)
        if df2.shape[1] > 1:
            ax = plt.pyplot.subplot(1,2,2)
            df2[df2[0]!=0].plot.bar(legend=False, fontsize=13, ax=ax)
            ax.set_title("Pred. changed from Not Default to Default")

        plt.pyplot.tight_layout()

        plt.pyplot.savefig("results/explanations/" + dataset_name + "/permuteattack_explanation_final_graph_" + model_name + ".png", dpi=1200)
        plt.pyplot.close("all")

'''
 Main method for SHAP/LIME explanations which aims to englobe the initialization of such methods, un-cluttering main.py
 model - sklearn predictive model, after .fit() method is applied to train data
 x_train/x_test - train/test data as numpy arrays
 cols - 
 target_values
 dataset_name - string representing the name of the dataset when saving the results
'''


# LIME explanation framework, takes four parameters as np.array and the black-box model
def lime_explainer(model, x_train, x_test, feature_labels, target_label, dataset_name, model_name, id=None):
    # if isinstance(x_train, np.float64)
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=feature_labels, class_names=target_label,
                                                       discretize_continuous=False)
    if id is None:
        np.random.randint(0, x_test.shape[0])

    exp = explainer.explain_instance(x_test[0], model.predict_proba, num_features=len(feature_labels))
    # exp.show_in_notebook(show_table=True, show_all=False)
    exp.save_to_file("results/explanations/" + dataset_name + "/lime_explanation_" + model_name + ".html")


# SHAP explanation framework, takes four parameters as np.array and the black-box model
def shap_explainer(model, x, feature_names, dataset_name, model_name, multioutput=False):
    shap.initjs()
    data = pd.DataFrame(x, columns=feature_names)
    # explainer = shap.KernelExplainer(model.predict_proba, data)
    plt.pyplot.clf()
    plt.pyplot.close("all")
# shap_values = explainer_tree.shap_values(data)
    if multioutput:
        explainer = shap.KernelExplainer(model.predict_proba, data)
        shap_values = explainer.shap_values(data)
        # fig = shap.waterfall(explainer_tree.expected_value[0], shap_values[0], x, show=False)
        shap.summary_plot(shap_values, data, class_names=model.classes_, show=False)
        plt.pyplot.savefig("results/explanations/" + dataset_name + "/shap_summary_" + model_name + ".png", dpi=1200, bbox_inches="tight")
        plt.pyplot.clf()
        plt.pyplot.close("all")
        # shap.waterfall_plot(explainer.expected_value[0], shap_values[0][0], show=False)
        shap.force_plot(explainer.expected_value[0], shap_values[0], data)
        plt.pyplot.savefig("results/explanations/" + dataset_name + "/shap_force_" + model_name + ".png", dpi=1200, bbox_inches="tight")
        plt.pyplot.clf()
        plt.pyplot.close("all")
    else:
        #masker = shap.maskers.Independent(data, 10)

        explainer = shap.Explainer(model.predict_proba, data)

        shap_values = explainer(data)
        #shap.plots.waterfall(shap_values[0], show=False)
        #plt.pyplot.savefig("results/explanations/" + dataset_name + "/shap_waterfall.png", bbox_inches="tight")
        plt.pyplot.clf()
        plt.pyplot.close("all")
        # fig = shap.force_plot(explainer_tree.expected_value, shap_values[0, :], x[0, :])
        # plt.pyplot.savefig("results/explanations/"+dataset_name+"/shap_force_plot.png")
        # plt.pyplot.clf()
        #
        #IMPORTANCIA DE APENAS UMA CLASSE, NAO SERVE BEM PARA MULTIOUTPUT
        shap.plots.bar(shap_values[:, :, 1], show=False)
        plt.pyplot.savefig("results/explanations/" + dataset_name + "/shap_bar_1_plot_" + model_name + ".png", dpi=1200, bbox_inches="tight")
        plt.pyplot.clf()

        shap.summary_plot(shap_values[:, :, 1], x, plot_type="bar", show=False)
        # fig = shap.summary_plot(shap_values.shap_values(data), data, plot_type="bar", show=False)
        plt.pyplot.savefig("results/explanations/" + dataset_name + "/shap_summary_plot_" + model_name + ".png", dpi=1200, bbox_inches="tight")
        plt.pyplot.clf()
        plt.pyplot.close("all")



        # Feature interactions is not useful...
        # fig = shap.summary_plot(shap_interactions, data, show=False)
        #for feat in feature_names:
        #    # fig = shap.plots.scatter(shap_values.shap_values(data)[:, feat], show=False)
        #    shap.plots.scatter(shap_values[:, feat, 1], show=False)
        #    plt.pyplot.savefig("results/explanations/" + dataset_name + "/shap_scatter_" + model_name + feat + ".png",
        #                       bbox_inches="tight")
        #    plt.pyplot.clf()

        # shap.plots.force(explainer.expected_value, shap_values[0], data.iloc[0,:], show=False)
        # plt.pyplot.savefig("results/explanations/"+dataset_name+"/shap_force.png")
        # plt.pyplot.clf()
        shap.plots.bar(shap_values[:, :, 1], show=False)
        plt.pyplot.savefig("results/explanations/" + dataset_name + "/shap_barplot_" + model_name + ".png", dpi=1200, bbox_inches="tight")
        plt.pyplot.clf()


def dice_explainer(train_dataset, x_test, model, continuous_features, target_name, classes, dataset_name, model_name):
    # Dataset for training an ML model
    d = models.DiCE.dice_ml.Data(dataframe=train_dataset,
                     continuous_features=continuous_features,
                     outcome_name=target_name)

    # Pre-trained ML model
    m = models.DiCE.dice_ml.Model(model=model, backend='sklearn')
    # DiCE explanation instance
    exp = models.DiCE.dice_ml.Dice(d, m)
    try:
        if len(classes) > 2:
            for target in classes:
                e1 = exp.generate_counterfactuals(train_dataset[0:1].drop([target_name], axis=1), total_CFs=7, desired_class=target)
                e1.visualize_as_dataframe()

        else:
            e1 = exp.generate_counterfactuals(x_test[0:1], total_CFs=3, desired_class="opposite",  features_to_vary=continuous_features)
            e1.cf_examples_list[0].test_instance_df
            e1.visualize_as_dataframe()


        final_data = []
        final_data.append("Original Instance\n")
        final_data.append(x_test.columns)
        final_data.append("\n")

        final_data.append(e1.cf_examples_list[0].test_instance_df.values)
        final_data.append("\nCounterfactuals Generated\n")
        final_data.append(e1.cf_examples_list[0].final_cfs_df.values)
        af.save_to_file_2("results/explanations/" + dataset_name + "/dice_counterfactuals_"+ model_name + ".txt", final_data)
        final_data = pd.DataFrame(e1.cf_examples_list[0].final_cfs_df)
        final_data = final_data.append(e1.cf_examples_list[0].test_instance_df)
        final_data.to_csv(path_or_buf='results/explanations/'+dataset_name+'/dice_counterfactuals_' + model_name + '.csv', index=False)
    except:
        print("No counterfactual explanations for given instance")

    #e1.cf_examples_list[0].final_cfs_df.to_csv(path_or_buf='results/explanations/'+dataset_name+'/dice_counterfactuals.csv', index=False)
