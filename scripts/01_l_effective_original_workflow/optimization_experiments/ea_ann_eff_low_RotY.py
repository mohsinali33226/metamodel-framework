import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize


TARGET = "eff_low_RotY"
CURRENT_BEST = 0.46611

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


def make_ann(n_layers, n1, n2, n3, n4, activation, lr, alpha):
    units = [n1, n2, n3, n4][:n_layers]
    ann = MLPRegressor(
        hidden_layer_sizes=tuple(units),
        activation=activation,
        solver="adam",
        learning_rate_init=lr,
        alpha=alpha,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=42,
        batch_size=64,
    )

    pipe = make_pipeline(StandardScaler(), ann)

    model = TransformedTargetRegressor(
        regressor=pipe,
        transformer=StandardScaler()
    )

    return model


class ANNProblem(ElementwiseProblem):
    def __init__(self):
        # n_layers, n1, n2, n3, n4, activation_code, log_lr, log_alpha
        super().__init__(
            n_var=8,
            n_obj=1,
            xl=np.array([1, 50, 50, 50, 50, 0, -4.0, -7.0]),
            xu=np.array([4, 500, 500, 500, 500, 1, -2.0, -3.0]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        n_layers = int(round(x[0]))
        n1 = int(round(x[1]))
        n2 = int(round(x[2]))
        n3 = int(round(x[3]))
        n4 = int(round(x[4]))

        activation = "relu" if int(round(x[5])) == 0 else "tanh"
        lr = 10 ** float(x[6])
        alpha = 10 ** float(x[7])

        model = make_ann(n_layers, n1, n2, n3, n4, activation, lr, alpha)

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


problem = ANNProblem()

algorithm = GA(
    pop_size=12,
    eliminate_duplicates=True,
)

res = minimize(
    problem,
    algorithm,
    termination=("n_gen", 8),
    seed=42,
    verbose=True,
)

x = res.X
best_cv_r2 = -float(res.F[0])

n_layers = int(round(x[0]))
params = {
    "n_layers": n_layers,
    "n1": int(round(x[1])),
    "n2": int(round(x[2])),
    "n3": int(round(x[3])),
    "n4": int(round(x[4])),
    "activation": "relu" if int(round(x[5])) == 0 else "tanh",
    "learning_rate_init": 10 ** float(x[6]),
    "alpha": 10 ** float(x[7]),
}

print("\nEA-ANN optimization result")
print("TARGET =", TARGET)
print("BEST_CV_R2 =", best_cv_r2)
print("BEST_PARAMS =", params)

final_model = make_ann(
    params["n_layers"],
    params["n1"],
    params["n2"],
    params["n3"],
    params["n4"],
    params["activation"],
    params["learning_rate_init"],
    params["alpha"],
)

final_model.fit(Xdev, ydev)
pred = final_model.predict(Xte)

test_r2 = r2_score(yte, pred)
test_mae = mean_absolute_error(yte, pred)

print("\nFinal hold-out result")
print("CURRENT_BEST =", CURRENT_BEST)
print("EA_ANN_TEST_R2 =", test_r2)
print("EA_ANN_TEST_MAE =", test_mae)
print("IMPROVED =", test_r2 > CURRENT_BEST)

out = "./data/outputs/l_effective/ea_ann_eff_low_RotY_result.csv"
pd.DataFrame([{
    "target": TARGET,
    "method": "pymoo_GA_ANN_MLP",
    "best_cv_r2": best_cv_r2,
    **params,
    "test_r2": test_r2,
    "test_mae": test_mae,
    "current_best": CURRENT_BEST,
    "improved": test_r2 > CURRENT_BEST,
}]).to_csv(out, index=False)

print("saved:", out)

