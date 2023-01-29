#permuteattack = importlib.import_module(".models.PermuteAttack-main.src")
#print(permuteattack)
#import permute.ga_attack
import models.PermuteAttack.src.ga_attack as permute

def permuteattack_explainer(model, feature_names, x_train, x_test, idx=0):
    permute_attack = permute.GAdvExample(feature_names=list(feature_names),
                             sol_per_pop=30, num_parents_mating=10, cat_vars_ohe=None,
                             num_generations=100, n_runs=10, black_list=[],
                             verbose=False, beta=.95)

    x_all, x_changes, x_sucess = permute_attack.attack(model, x=x_test[idx, :], x_train=x_train)
    print("X_ALL \n", x_all, "\n", "X_CHANGES", "\n", x_changes, "\n", "X_SUCCESS", "\n", x_sucess)

