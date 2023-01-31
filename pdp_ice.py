from sklearn.inspection import PartialDependenceDisplay, partial_dependence
import matplotlib.pyplot as plt
import numpy as np

def pdp_explainer(model, x_axis, features, feature_names, dataset_name):
    pdp_results = partial_dependence(model, x_axis, features)
    pdp = PartialDependenceDisplay.from_estimator(model, x_axis, features)
    deciles = {0: np.linspace(0, 1, num=5)}
    #pdp = PartialDependenceDisplay([pdp_results], target_idx=0, features=features, feature_names=feature_names, deciles=deciles)
    plt.savefig("results/explanations/"+dataset_name+"/"+str(features[0])+"_pdp_explanation.png")
    plt.show()
#pdp.plot().show()
    #fig = plt.figure()
        #pdp.plot()#.gcf()
    #pdp.plot().gca()
    #return fig
