import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize


TARGET = "eff_high_RotZ"
CURRENT_BEST = 0.66674

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


class XGBProblem(ElementwiseProblem):
    def __init__(self):
        # variables:
        # max_depth, n_estimators, learning_rate, subsample,
        # colsample_bytree, reg_lambda, min_child_weight
        super().__init__(
            n_var=7,
            n_obj=1,
            xl=np.array([2, 200, 0.005, 0.5, 0.5, 0.001, 0.001]),
            xu=np.array([10, 1200, 0.08, 1.0, 1.0, 20.0, 10.0]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        max_depth = int(round(x[0]))
        n_estimators = int(round(x[1]))
        learning_rate = float(x[2])
        subsample = float(x[3])
        colsample_bytree = float(x[4])
        reg_lambda = float(x[5])
        min_child_weight = float(x[6])

        model = XGBRegressor(
            objective="reg:squarederror",
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
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


problem = XGBProblem()

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
    "learning_rate": float(x[2]),
    "subsample": float(x[3]),
    "colsample_bytree": float(x[4]),
    "reg_lambda": float(x[5]),
    "min_child_weight": float(x[6]),
}

print("\nEA-XGBoost optimization result")
print("TARGET =", TARGET)
print("BEST_CV_R2 =", best_cv_r2)
print("BEST_PARAMS =", params)

final_model = XGBRegressor(
    objective="reg:squarederror",
    **params,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

final_model.fit(Xdev, ydev)
pred = final_model.predict(Xte)

test_r2 = r2_score(yte, pred)
test_mae = mean_absolute_error(yte, pred)

print("\nFinal hold-out result")
print("CURRENT_BEST =", CURRENT_BEST)
print("EA_XGB_TEST_R2 =", test_r2)
print("EA_XGB_TEST_MAE =", test_mae)
print("IMPROVED =", test_r2 > CURRENT_BEST)

out = "./data/outputs/l_effective/ea_xgb_eff_high_RotZ_result.csv"
pd.DataFrame([{
    "target": TARGET,
    "method": "pymoo_GA_XGBoost",
    "best_cv_r2": best_cv_r2,
    **params,
    "test_r2": test_r2,
    "test_mae": test_mae,
    "current_best": CURRENT_BEST,
    "improved": test_r2 > CURRENT_BEST,
}]).to_csv(out, index=False)

print("saved:", out)

