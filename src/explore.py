import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("data/ai4i2020.csv")


"""
#donne les 5 premiers lignes (pour avoir une vision sur les lignes) :
print(df.head())

#donne (nbr_lignes, nbr_colonnes):
print(df.shape)

#donne les colonnes notre dataframe :
print(df.columns)

#donne type de chaque colonnes:
print(df.dtypes)

#donne des infos generale comme colonnes, nbr items de chaque colonne et son type, nbr de lignes :
df.info()

#donne count, mouyenne, ecart-type, min/max et les quartiles :
print(df.describe())


#donne l occurence de chaque valeur dans la colonne :
print(df["Machine failure"].value_counts())

#en portion :
print(df["Machine failure"].value_counts(normalize=True))

#donne la moyenne de ces colonnes dans les cas possibles de 'Machine failure' (0, 1) :
print(
    df.groupby("Machine failure")[[
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]].mean()
)

#occurence couple type/machine_failure :
print(
    df.groupby(["Type", "Machine failure"]).size()
)

#normalement donne la moyenne machine_failure de chaque type et puisque machine_failure soit 0 soit 1, donc cela donne pourcentage de panne dans chaque type :
print(
    df.groupby("Type")["Machine failure"].mean()
)

#le nombre de valeurs manquantes par colonne :
print(df.isna().sum())

#le nombre de doublons :
print(df.duplicated().sum())


'''
isna()             → valeurs manquantes
duplicated()        → doublons
loc                 → sélection précise
&                   → combiner des conditions
nouvelle colonne    → feature engineering
sort_values()       → classement
np.where()          → transformation conditionnelle
'''



"""

#normalisation des colonnes:
df = df.rename(columns={
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf"
})

# print(df.columns)

# 1er feature :
df["temperature_difference"] = (
    df["process_temperature"]
    - df["air_temperature"]
)



#  Visualisation des donnes avec matplotlib :

""" y a une grande difference entre le nbr de normal_torque et failure_torque so c est mieux de passer a la comparaison de densite :
normal_tool_wear = df.loc[
    df["machine_failure"] == 0,
    "tool_wear"
]

failure_tool_wear = df.loc[
    df["machine_failure"] == 1,
    "tool_wear"
]
plt.hist(
    normal_torque,
    bins=30,
    alpha=0.5,
    label="Normal"
)

plt.hist(
    failure_torque,
    bins=30,
    alpha=0.5,
    label="Failure"
)

plt.xlabel("Torque [Nm]")
plt.ylabel("Nombre d'observations")
plt.title("Torque : fonctionnement normal vs panne")
plt.legend()

plt.show()
"""

"""
# comparaison densite :
plt.hist(
    normal_tool_wear,
    bins=30,
    alpha=0.5,
    density=True,
    label="Normal"
)

plt.hist(
    failure_tool_wear,
    bins=30,
    alpha=0.5,
    density=True,
    label="Failure"
)

plt.xlabel("tool_wear [min]")
plt.ylabel("Densité")
plt.title("Distribution du tool_wear : normal vs panne")
plt.legend()

plt.show()
"""


#data quality verification :
#print(df.isna().sum())
#print(df.duplicated().sum())



# fonction de dessin des histogrammes de densite de chaque feature dans les deux cas (normal et failure):
def plot_feature_distribution(df, feature, unit=""):
    normal=df.loc[
        df["machine_failure"] == 0,
        feature
    ]

    failure=df.loc[
        df["machine_failure"] == 1,
        feature
    ]

    plt.hist(
        normal,
        bins=30,
        alpha=0.5,
        density=True,
        label="Normal"
    )

    plt.hist(
        failure,
        bins=30,
        alpha=0.5,
        density=True,
        label="Failure"
    )

    plt.xlabel(f"{feature} {unit}")
    plt.ylabel("densité")
    plt.title(f"Distribution de {feature} : normal vs panne")

    plt.show()

"""
#plot_feature_distribution(df, "torque", "[Nm]")
#plot_feature_distribution(df, "tool_wear", "[min]")
plot_feature_distribution(df, "rotational_speed", "[rpm]")
plot_feature_distribution(df, "air_temperature", "[K]")
plot_feature_distribution(df, "process_temperature", "[K]")
plot_feature_distribution(df, "temperature_difference", "[K]")
"""

features = [
    "type",
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "temperature_difference"
]

x = df[features]

y = df["machine_failure"]

x = pd.get_dummies(
    x,
    columns=["type"],
    dtype=int
)

"""
print(x.head())
print(x.columns)
print(x.shape)
"""

# partie de split dataset train/test :
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42, #pour avoir les meme donnes si on refait le spli
    stratify=y #pour equilibrer les cas de y entre test et train
)


"""
      # verification de split :
print("X_train :", x_train.shape)
print("X_test  :", x_test.shape)

print("\nRépartition train :")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

print("\nRépartition test :")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True))
"""



#construction et verification de baseline :
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

y_pred_baseline = np.zeros(len(y_test), dtype=int)

cm = confusion_matrix(y_test, y_pred_baseline)

accuracy = accuracy_score(y_test, y_pred_baseline)

precision = precision_score(
    y_test,
    y_pred_baseline,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred_baseline
)

f1 = f1_score(
    y_test,
    y_pred_baseline
)

print("Confusion Matrix :", cm, "\nAccuracy :", accuracy, "\nprecision :", precision, "\nRecall :", recall, "\nF1-acore :", f1) 

