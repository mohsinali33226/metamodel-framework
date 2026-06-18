import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize


TARGET = "eff_broad_Z"
CURRENT_BEST = 0.78698

Xtr = pd.read_csv("./data/outputs/l_effective/X_train.csv")
Xva = pd.read_csv("./data/outputs/l_effective/X_val.csv")
Xte = pd.read_csv("./data/outputs/l_effective/X_test.csv")

Ytr = pd.read_csv("./data/outputs/l_effective/Y_train.csv")
Yva = pd.read_csv("./data/outputs/l_effective/Y_val.csv")
Yte = pd.read_csv("./data/outputs/l_effective/Y_test.csv")

Xdev = pd.concat([Xtr, Xva], axis=0).reset_index(drop=True)
ydev = pd.concat([Ytr[TARGET], Yva[TARGET]], axis=0).reset_index(drop=True)
yte = Yte[TARGET]

cv = KFold(n_splits=5, shuffle=True, random_state=42)


class SVRHyperparameterProblem(ElementwiseProblem):
    def __init__(self):
        # variables are log10(C), log10(epsilon), log10(gamma)
        super().__init__(
            n_var=3,
            n_obj=1,
            xl=np.array([1.0, -3.0, -3.0]),
            xu=np.array([4.0, 0.5, 0.0]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        C = 10 ** x[0]
        epsilon = 10 ** x[1]
        gamma = 10 ** x[2]

        model = make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma),
        )

        scores = cross_val_score(
            model,
            Xdev,
            ydev,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
        )

        # pymoo minimizes, so minimize negative mean R2
        out["F"] = -float(scores.mean())


problem = SVRHyperparameterProblem()

algorithm = GA(
    pop_size=20,
    eliminate_duplicates=True,
)

res = minimize(
    problem,
    algorithm,
    termination=("n_gen", 15),
    seed=42,
    verbose=True,
)

best_x = res.X
best_cv_r2 = -float(res.F[0])

best_C = 10 ** best_x[0]
best_epsilon = 10 ** best_x[1]
best_gamma = 10 ** best_x[2]

print("\nEA optimization result")
print("TARGET =", TARGET)
print("BEST_CV_R2 =", best_cv_r2)
print("BEST_C =", best_C)
print("BEST_EPSILON =", best_epsilon)
print("BEST_GAMMA =", best_gamma)

final_model = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=best_C, epsilon=best_epsilon, gamma=best_gamma),
)
final_model.fit(Xdev, ydev)
pred = final_model.predict(Xte)

test_r2 = r2_score(yte, pred)
test_mae = mean_absolute_error(yte, pred)

print("\nFinal hold-out result")
print("CURRENT_BEST =", CURRENT_BEST)
print("EA_TEST_R2 =", test_r2)
print("EA_TEST_MAE =", test_mae)
print("IMPROVED =", test_r2 > CURRENT_BEST)

out = "./data/outputs/l_effective/ea_svr_eff_broad_Z_result.csv"
pd.DataFrame([{
    "target": TARGET,
    "method": "pymoo_GA_SVR",
    "best_cv_r2": best_cv_r2,
    "C": best_C,
    "epsilon": best_epsilon,
    "gamma": best_gamma,
    "test_r2": test_r2,
    "test_mae": test_mae,
    "current_best": CURRENT_BEST,
    "improved": test_r2 > CURRENT_BEST,
}]).to_csv(out, index=False)

print("saved:", out)

