import importlib
import numpy as np
import sys
#sys.path.append("/models/anchor")
#anchor = importlib.import_module("models.anchor-master.anchor")
import models.anchor.anchor.utils as anchor_utils
import models.anchor.anchor.anchor_tabular as anchor_tabular
from models.anchor.anchor.utils import map_array_values


def anchor_explainer(model, dataset, class_name, feature_names, features_to_use, categorical_features):#, , categorical_names, idx=0):#train, test, train_label, test_label, class_name, feature_names, categorical_names, idx=0):

    dataset_anchors = anchor_utils.load_csv_dataset(dataset, features_to_use=features_to_use, feature_names=feature_names, categorical_features=categorical_features,target_idx=class_name, skip_first=True)


    explainer = anchor_tabular.AnchorTabularExplainer(
        dataset_anchors.class_names,
        dataset_anchors.feature_names,
        np.array(dataset_anchors.train),
        dataset_anchors.categorical_names)
    idx = 0
    np.random.seed(1)
    #print('Prediction: ', explainer.class_names[model.predict(test[idx].reshape(1, -1))[0]])
    exp = explainer.explain_instance(np.array(dataset_anchors.test)[idx,:], model.predict, threshold=0.95)

    print('Anchor: %s' % (' AND '.join(exp.names())))
    print('Precision: %.2f' % exp.precision())
    print('Coverage: %.2f' % exp.coverage())

    # Get test examples where the anchora pplies
    fit_anchor = np.where(np.all(np.array(dataset_anchors.test)[:, exp.features()] == np.array(dataset_anchors.test)[idx][exp.features()], axis=1))[0]
    print('Anchor test precision: %.2f' % (np.mean(model.predict(np.array(dataset_anchors.test)[fit_anchor]) == model.predict(np.array(dataset_anchors.test)[idx].reshape(1, -1)))))
    print('Anchor test coverage: %.2f' % (fit_anchor.shape[0] / float(np.array(dataset_anchors.test).shape[0])))


    print('Partial anchor: %s' % (' AND '.join(exp.names(1))))
    print('Partial precision: %.2f' % exp.precision(1))
    print('Partial coverage: %.2f' % exp.coverage(1))


    fit_partial = np.where(np.all(np.array(dataset_anchors.test)[:, exp.features(1)] == np.array(dataset_anchors.test)[idx][exp.features(1)], axis=1))[0]
    print('Partial anchor test precision: %.2f' % (np.mean(model.predict(np.array(dataset_anchors.test)[fit_partial]) == model.predict(np.array(dataset_anchors.test)[idx].reshape(1, -1)))))
    print('Partial anchor test coverage: %.2f' % (fit_partial.shape[0] / float(np.array(dataset_anchors.test).shape[0])))