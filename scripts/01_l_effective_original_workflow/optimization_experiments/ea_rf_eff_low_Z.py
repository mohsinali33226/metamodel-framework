import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize


TARGET = "eff_low_Y"
CURRENT_BEST = 0.52974

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


class RFProblem(ElementwiseProblem):
    def __init__(self):
        # max_depth, n_estimators, min_samples_leaf, max_features
        super().__init__(
            n_var=4,
            n_obj=1,
            xl=np.array([2, 200, 1, 0.3]),
            xu=np.array([80, 1200, 10, 1.0]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        max_depth = int(round(x[0]))
        n_estimators = int(round(x[1]))
        min_samples_leaf = int(round(x[2]))
        max_features = float(x[3])

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )

        scores = cross_val_score(
            model,
            Xdev,
            ydev,
            cv=cv,
            scoring="r2",
            n_jobs=1,
        )

        out["F"] = -float(scores.mean())


problem = RFProblem()

algorithm = GA(
    pop_size=25,
    eliminate_duplicates=True,
)

res = minimize(
    problem,
    algorithm,
    termination=("n_gen", 12),
    seed=42,
    verbose=True,
)

x = res.X
best_cv_r2 = -float(res.F[0])

params = {
    "max_depth": int(round(x[0])),
    "n_estimators": int(round(x[1])),
    "min_samples_leaf": int(round(x[2])),
    "max_features": float(x[3]),
}

print("\nEA-RandomForest optimization result")
print("TARGET =", TARGET)
print("BEST_CV_R2 =", best_cv_r2)
print("BEST_PARAMS =", params)

final_model = RandomForestRegressor(
    **params,
    random_state=42,
    n_jobs=-1,
)

final_model.fit(Xdev, ydev)
pred = final_model.predict(Xte)

test_r2 = r2_score(yte, pred)
test_mae = mean_absolute_error(yte, pred)

print("\nFinal hold-out result")
print("CURRENT_BEST =", CURRENT_BEST)
print("EA_RF_TEST_R2 =", test_r2)
print("EA_RF_TEST_MAE =", test_mae)
print("IMPROVED =", test_r2 > CURRENT_BEST)

out = "./data/outputs/l_effective/ea_rf_eff_low_Y_result.csv"
pd.DataFrame([{
    "target": TARGET,
    "method": "pymoo_GA_RandomForest",
    "best_cv_r2": best_cv_r2,
    **params,
    "test_r2": test_r2,
    "test_mae": test_mae,
    "current_best": CURRENT_BEST,
    "improved": test_r2 > CURRENT_BEST,
}]).to_csv(out, index=False)

print("saved:", out)

