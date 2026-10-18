import joblib
import pandas as pd
import time 

from data_processing import prepare_features 

# recuperation de 100 arbres deja entreines :
model = joblib.load("models/random_forest.pkl")




data = pd.read_csv("data/test_data.csv")

normal_samples = data[
    data["Machine failure"] == 0
].sample(
    n=10,
    random_state=42
)

failure_samples = data[
    data["Machine failure"] == 1
].sample(
    n=10,
    random_state=42
)

stream_data = pd.concat([
    normal_samples,
    failure_samples
]).sample(
    frac=1,
    random_state=42
)




threshold = 0.6

for _, machine in stream_data.iterrows():

    machine_df = pd.DataFrame([machine])

    machine_features = prepare_features(machine_df)

    failure_probability = model.predict_proba(
        machine_features
    )[0, 1]

    real_status = machine["Machine failure"]

    print("\n-----------------------------")

    print(f"Machine: {machine['Product ID']}")
    print(f"Type: {machine['Type']}")
    print(f"Torque: {machine['Torque [Nm]']} Nm")
    print(f"Tool wear: {machine['Tool wear [min]']} min")

    print(
    f"Real status: "
    f"{'FAILURE' if real_status == 1 else 'NORMAL'}"
)

    print(
        f"Failure risk: "
        f"{failure_probability:.2%}"
    )

    if failure_probability >= threshold:
        print("⚠ MAINTENANCE ALERT")
    else:
        print("✓ NORMAL")

    time.sleep(1)