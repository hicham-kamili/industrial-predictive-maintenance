import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import RandomForestClassifier 

import joblib

from data_processing import prepare_features 


# 1. Charger les données
df = pd.read_csv("data/ai4i2020.csv")

df_raw = df.copy()

X = prepare_features(df)
y = df["Machine failure"]


# 6. Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# conserver data de test apart pour le simulator :
test_index = X_test.index
df_raw.loc[test_index].to_csv(
    "data/test_data.csv",
    index=False
)

# 7. Baseline
y_pred_baseline = np.zeros(len(y_test), dtype=int)


# 8. Evaluation
print("=== BASELINE ===")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_baseline))

print("Accuracy :", accuracy_score(y_test, y_pred_baseline))

print(
    "Precision:",
    precision_score(
        y_test,
        y_pred_baseline,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(y_test, y_pred_baseline)
)

print(
    "F1-score:",
    f1_score(y_test, y_pred_baseline)
)

print("----------------------------------------------------")




#training regression Logistic model ;
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== LOGISTIC REGRESSION ===")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy :", accuracy_score(y_test, y_pred))

print(
    "Precision:",
    precision_score(y_test, y_pred, zero_division=0)
)

print(
    "Recall:",
    recall_score(y_test, y_pred)
)

print(
    "F1-score:",
    f1_score(y_test, y_pred)
)


# Logistic Regression balanced :

balanced_model = Pipeline([
    ("scaler", StandardScaler()), #pour standariser les variables (features doivent etre meme echelle)
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ))
])

balanced_model.fit(X_train, y_train)

y_pred_balanced = balanced_model.predict(X_test)

print("\n=== LOGISTIC REGRESSION BALANCED ===")

print(
    "Confusion Matrix: \n",
    confusion_matrix(y_test, y_pred_balanced),
    "\nAccuracy: ",
    accuracy_score(y_test, y_pred_balanced),
    "\nPrecision: ",
    precision_score(y_test, y_pred_balanced, zero_division=0),
    "\nRecall: ",
    recall_score(y_test, y_pred_balanced),
    "\nF1-score: ",
    f1_score(y_test, y_pred_balanced)
    )
     



# passons maintenant au model Rnadom Forest :

rf_model = RandomForestClassifier(
    n_estimators=100, #nbr des arbres
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

# sérialiser le modèle : pour entrainer le model une seul fois et le conserver pour la prediction de new entrees
joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)

print("\nModel saved: models/random_forest.pkl")


y_pred_rf = rf_model.predict(X_test)

print("\n=== RANDOM FOREST BALANCED ===")

print(
    "Confusion Matrix: \n",
    confusion_matrix(y_test, y_pred_rf),
    "\nAccuracy: ",
    accuracy_score(y_test, y_pred_rf),
    "\nPrecision: ",
    precision_score(y_test, y_pred_rf, zero_division=0),
    "\nRecall: ",
    recall_score(y_test, y_pred_rf),
    "\nF1-score: ",
    f1_score(y_test, y_pred_rf)
    )

# Testons plusieurs :
failure_proba = rf_model.predict_proba(X_test)[:, 1]

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

for threshold in thresholds:

    y_pred_threshold = (
        failure_proba >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred_threshold
    )

    f1 = f1_score(
        y_test,
        y_pred_threshold
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")