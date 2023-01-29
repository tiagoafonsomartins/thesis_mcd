import importlib
import sys
sys.path.append("C:/Users/Rhinestein/Documents/ISCTE/Código Tese/thesis_mcd/models/PermuteAttack-main/src")
sys.path.append("C:/Users/Rhinestein/Documents/ISCTE/Código Tese/thesis_mcd/models/anchor-master")
anchor = importlib.import_module("models.anchor-master.anchor")
permuteattack = importlib.import_module(".models.PermuteAttack-main.src")
print(permuteattack)
#import permute.ga_attack


#def permuteattack_explainer(model, x_train, x_test, idx=0):
#    ga_titanic = permute.GAdvExample(feature_names=list(x_train.columns),
#                             sol_per_pop=30, num_parents_mating=10, cat_vars_ohe=None,
#                             num_generations=100, n_runs=10, black_list=[],
#                             verbose=False, beta=.95)
#
#    x_all, x_changes, x_sucess = ga_titanic.attack(model, x=x_test[idx, :], x_train=x_train)
#    print("X_ALL \n", x_all, "\n", "X_CHANGES", "\n", x_changes, "\n", "X_SUCCESS", "\n", x_sucess)

