import importlib
import numpy as np
import sys
sys.path.append("C:/Users/Rhinestein/Documents/ISCTE/Código Tese/thesis_mcd\models/anchor-master")
anchor = importlib.import_module("models.anchor-master.anchor")
import anchor.utils
import anchor.anchor_tabular
from anchor.utils import map_array_values


def anchor_explanation(model, dataset_train, dataset_test, class_name, feature_names, categorical_names, id=0):

    explainer = anchor.anchor_tabular.AnchorTabularExplainer(
        class_name,
        feature_names,
        dataset_train,
        categorical_names)

    print('Prediction: ', explainer.class_names[model.predict(dataset_test[id].reshape(1, -1))[0]])
    exp = explainer.explain_instance(dataset_test[id], model.predict, threshold=0.95)

    print('Anchor: %s' % (' AND '.join(exp.names())))
    print('Precision: %.2f' % exp.precision())
    print('Coverage: %.2f' % exp.coverage())

    # Get test examples where the anchora pplies
    fit_anchor = np.where(np.all(dataset_test[:, exp.features()] == dataset_test[id][exp.features()], axis=1))[0]
    print('Anchor test precision: %.2f' % (np.mean(model.predict(dataset_test[fit_anchor]) == model.predict(dataset_test[id].reshape(1, -1)))))
    print('Anchor test coverage: %.2f' % (fit_anchor.shape[0] / float(dataset_test.shape[0])))


    print('Partial anchor: %s' % (' AND '.join(exp.names(1))))
    print('Partial precision: %.2f' % exp.precision(1))
    print('Partial coverage: %.2f' % exp.coverage(1))


    fit_partial = np.where(np.all(dataset_test[:, exp.features(1)] == dataset_test[id][exp.features(1)], axis=1))[0]
    print('Partial anchor test precision: %.2f' % (np.mean(model.predict(dataset_test[fit_partial]) == model.predict(dataset_test[id].reshape(1, -1)))))
    print('Partial anchor test coverage: %.2f' % (fit_partial.shape[0] / float(dataset_test.shape[0])))