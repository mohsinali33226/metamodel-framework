import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize


TARGET = "eff_low_RotY"
CURRENT_BEST = 0.46770

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(1)

Xtr = pd.read_csv("./data/outputs/l_effective/X_train.csv")
Xva = pd.read_csv("./data/outputs/l_effective/X_val.csv")
Xte = pd.read_csv("./data/outputs/l_effective/X_test.csv")

Ytr = pd.read_csv("./data/outputs/l_effective/Y_train.csv")
Yva = pd.read_csv("./data/outputs/l_effective/Y_val.csv")
Yte = pd.read_csv("./data/outputs/l_effective/Y_test.csv")

Xdev = pd.concat([Xtr, Xva], axis=0).reset_index(drop=True).values.astype(np.float32)
ydev = pd.concat([Ytr[TARGET], Yva[TARGET]], axis=0).reset_index(drop=True).values.reshape(-1, 1).astype(np.float32)

Xtest = Xte.values.astype(np.float32)
ytest = Yte[TARGET].values.reshape(-1, 1).astype(np.float32)

cv = KFold(n_splits=5, shuffle=True, random_state=42)


class ANN(nn.Module):
    def __init__(self, input_dim, n_layers, neurons, activation_name):
        super().__init__()
        layers = []
        prev = input_dim

        for i in range(n_layers):
            layers.append(nn.Linear(prev, neurons[i]))

            if activation_name == "relu":
                layers.append(nn.ReLU())
            elif activation_name == "elu":
                layers.append(nn.ELU())
            else:
                layers.append(nn.LeakyReLU(negative_slope=0.1))

            prev = neurons[i]

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_and_predict(X_train, y_train, X_val, n_layers, neurons, activation, lr, epochs=250):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = x_scaler.transform(X_val).astype(np.float32)

    y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)

    X_train_t = torch.tensor(X_train_s)
    y_train_t = torch.tensor(y_train_s)
    X_val_t = torch.tensor(X_val_s)

    model = ANN(X_train_s.shape[1], n_layers, neurons, activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    batch_size = 64
    n = X_train_t.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n)

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)

            if not torch.isfinite(loss):
                raise ValueError("non-finite loss")

            loss.backward()
            optimizer.step()

    with torch.no_grad():
        pred_s = model(X_val_t).numpy()

    pred = y_scaler.inverse_transform(pred_s).ravel()
    return pred


class PaperANNProblem(ElementwiseProblem):
    def __init__(self):
        # Paper Table 6 ANN:
        # neurons layer 1â€“4: 100â€“600
        # learning rate: 0.0001â€“0.9
        # activation: ELU, ReLU, LeakyReLU
        super().__init__(
            n_var=7,
            n_obj=1,
            xl=np.array([1, 100, 100, 100, 100, 0.0001, 0]),
            xu=np.array([4, 600, 600, 600, 600, 0.9, 2]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        n_layers = int(round(x[0]))
        neurons = [
            int(round(x[1])),
            int(round(x[2])),
            int(round(x[3])),
            int(round(x[4])),
        ]

        lr = float(x[5])
        activation = ["elu", "relu", "leaky_relu"][max(0, min(2, int(round(x[6]))))]

        scores = []

        try:
            for tr_idx, va_idx in cv.split(Xdev):
                pred = train_and_predict(
                    Xdev[tr_idx],
                    ydev[tr_idx],
                    Xdev[va_idx],
                    n_layers,
                    neurons,
                    activation,
                    lr,
                    epochs=250,
                )

                score = r2_score(ydev[va_idx].ravel(), pred)
                scores.append(score)

            final_score = float(np.mean(scores))

        except Exception:
            final_score = -999.0

        out["F"] = -final_score


problem = PaperANNProblem()

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
    "n_layers": int(round(x[0])),
    "n1": int(round(x[1])),
    "n2": int(round(x[2])),
    "n3": int(round(x[3])),
    "n4": int(round(x[4])),
    "learning_rate": float(x[5]),
    "activation": ["elu", "relu", "leaky_relu"][max(0, min(2, int(round(x[6]))))],
}

print("\nPaper-style PyTorch ANN optimization result")
print("TARGET =", TARGET)
print("BEST_CV_R2 =", best_cv_r2)
print("BEST_PARAMS =", params)

neurons = [params["n1"], params["n2"], params["n3"], params["n4"]]

pred = train_and_predict(
    Xdev,
    ydev,
    Xtest,
    params["n_layers"],
    neurons,
    params["activation"],
    params["learning_rate"],
    epochs=400,
)

test_r2 = r2_score(ytest.ravel(), pred)
test_mae = mean_absolute_error(ytest.ravel(), pred)

print("\nFinal hold-out result")
print("CURRENT_BEST =", CURRENT_BEST)
print("PAPER_EA_TORCH_ANN_TEST_R2 =", test_r2)
print("PAPER_EA_TORCH_ANN_TEST_MAE =", test_mae)
print("IMPROVED =", test_r2 > CURRENT_BEST)

out = "./data/outputs/l_effective/paper_ea_torch_ann_eff_low_RotY_result.csv"
pd.DataFrame([{
    "target": TARGET,
    "method": "paper_style_EA_PyTorch_ANN",
    "best_cv_r2": best_cv_r2,
    **params,
    "test_r2": test_r2,
    "test_mae": test_mae,
    "current_best": CURRENT_BEST,
    "improved": test_r2 > CURRENT_BEST,
}]).to_csv(out, index=False)

print("saved:", out)

