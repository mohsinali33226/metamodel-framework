import joblib

# Change this line whenever you want to inspect another saved model
model_path = r"data/outputs/l_effective/models/GP_RBF__eff_high_RotY.joblib"

model = joblib.load(model_path)

print("Model file:")
print(model_path)

print("\nModel loaded successfully")
print("Type of object:", type(model))

print("\nObject itself:")
print(model)

if hasattr(model, "named_steps"):
    print("\nPipeline steps:")
    for name, step in model.named_steps.items():
        print(f"\n{name} -> {type(step)}")
        print(step)

        if hasattr(step, "get_params"):
            print("\nMain parameters:")
            params = step.get_params()
            for k, v in params.items():
                print(f"{k} = {v}")

