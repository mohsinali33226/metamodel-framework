import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize


TARGET = "eff_low_RotY"
CURRENT_BEST = 0.46770

Xtr = pd.read_csv("./data/outputs/l_effective/X_train.csv")
Xva = pd.read_csv("./data/outputs/l_effective/X_val.csv")
Xte = pd.read_csv("./data/outputs/l_effective/X_test.csv")

Ytr = pd.read_csv("./data/outputs/l_effective/Y_train.csv")
Yva = pd.read_csv("./data/outputs/l_effective/Y_val.csv")
Yte = pd.read_csv("./data/outputs/l_effective/Y_test.csv")

Xdev = pd.concat([Xtr, Xva], axis=0).reset_index(drop=True)
ydev = pd.concat([Ytr[TARGET], Yva[TARGET]], axis=0).reset_index(drop=True)
yte = Yte[TARGET]

n_features = Xdev.shape[1]
cv = KFold(n_splits=5, shuffle=True, random_state=42)


class PaperRFProblem(ElementwiseProblem):
    def __init__(self):
        # Paper Table 6 Random Forest hyperparameters:
        # max_depth: 20â€“200
        # n_estimators: 100â€“1500
        # min_samples_leaf: 2â€“10
        # max_features: 10â€“17, clipped to available features
        # max_leaf_nodes: 10â€“500
        # ccp_alpha / pruning: 0â€“0.5
        super().__init__(
            n_var=6,
            n_obj=1,
            xl=np.array([20, 100, 2, 10, 10, 0.0]),
            xu=np.array([200, 1500, 10, min(17, n_features), 500, 0.5]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        max_depth = int(round(x[0]))
        n_estimators = int(round(x[1]))
        min_samples_leaf = int(round(x[2]))
        max_features = int(round(x[3]))
        max_leaf_nodes = int(round(x[4]))
        ccp_alpha = float(x[5])

        max_features = max(1, min(max_features, n_features))

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            ccp_alpha=ccp_alpha,
            random_state=42,
            n_jobs=-1,
        )

        try:
            scores = cross_val_score(
                model,
                Xdev,
                ydev,
                cv=cv,
                scoring="r2",
                n_jobs=1,
            )
            score = float(scores.mean())
        except Exception:
            score = -999.0

        out["F"] = -score


problem = PaperRFProblem()

algorithm = GA(
    pop_size=50,
    crossover=SBX(prob=0.8, eta=15),
    mutation=PM(prob=0.15, eta=20),
    eliminate_duplicates=True,
)

res = minimize(
    problem,
    algorithm,
    termination=("n_gen", 30),
    seed=42,
    verbose=True,
)

x = res.X
best_cv_r2 = -float(res.F[0])

params = {
    "max_depth": int(round(x[0])),
    "n_estimators": int(round(x[1])),
    "min_samples_leaf": int(round(x[2])),
    "max_features": int(round(x[3])),
    "max_leaf_nodes": int(round(x[4])),
    "ccp_alpha": float(x[5]),
}

params["max_features"] = max(1, min(params["max_features"], n_features))

print("\nPaper-style EA-RandomForest optimization result")
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
print("PAPER_EA_RF_TEST_R2 =", test_r2)
print("PAPER_EA_RF_TEST_MAE =", test_mae)
print("IMPROVED =", test_r2 > CURRENT_BEST)

out = "./data/outputs/l_effective/paper_ea_rf_eff_low_RotY_result.csv"
pd.DataFrame([{
    "target": TARGET,
    "method": "paper_style_EA_RandomForest",
    "best_cv_r2": best_cv_r2,
    **params,
    "test_r2": test_r2,
    "test_mae": test_mae,
    "current_best": CURRENT_BEST,
    "improved": test_r2 > CURRENT_BEST,
}]).to_csv(out, index=False)

print("saved:", out)

