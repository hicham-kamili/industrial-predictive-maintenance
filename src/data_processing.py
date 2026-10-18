import pandas as pd


FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "temperature_difference",
    "type_H",
    "type_L",
    "type_M"
]

def prepare_features(df):

    df = df.copy()

    # Rename raw AI4I columns
    df = df.rename(columns={
        "Type": "type",
        "Air temperature [K]": "air_temperature",
        "Process temperature [K]": "process_temperature",
        "Rotational speed [rpm]": "rotational_speed",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear"
    })

    # Feature engineering
    df["temperature_difference"] = (
        df["process_temperature"]
        - df["air_temperature"]
    )

    # One-Hot Encoding
    df["type_H"] = (df["type"] == "H").astype(int)
    df["type_L"] = (df["type"] == "L").astype(int)
    df["type_M"] = (df["type"] == "M").astype(int)

    return df[FEATURES]