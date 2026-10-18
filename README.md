# Industrial Predictive Maintenance Platform

An end-to-end Data & AI project for predicting industrial equipment failures from machine operating data.

The project currently implements a machine-learning pipeline from raw industrial data to failure prediction and simulated machine monitoring. It is designed to progressively evolve toward a real-time Data Engineering and Cloud architecture.

## 🎯 Project Objective

Unexpected equipment failures can lead to production downtime, maintenance costs and operational disruptions.

The objective of this project is to use machine operating data to estimate the risk of equipment failure and generate maintenance alerts before a failure occurs.

The current version implements:

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Imbalanced classification analysis
- Logistic Regression baseline models
- Random Forest classification
- Failure probability estimation
- Simulated sequential machine monitoring
- Maintenance alert generation

## 📊 Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

The dataset contains **10,000 observations** representing synthetic industrial machine operating conditions.

Main variables include:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Product type
- Machine failure

The target variable is:

`Machine failure`

The dataset is strongly imbalanced:

- Normal operation: **96.61%**
- Machine failure: **3.39%**

This makes accuracy alone unsuitable for evaluating the predictive model.

The failure-mode columns `TWF`, `HDF`, `PWF`, `OSF` and `RNF` are deliberately excluded from the model features to avoid target leakage.

## 🔎 Exploratory Data Analysis

EDA showed several differences between normal and failure observations.

In particular:

- failures tend to occur with higher torque values;
- tool wear is generally higher for failed machines;
- rotational speed presents nonlinear failure patterns;
- temperature-related variables contain useful but overlapping information.

A derived feature was therefore introduced:

```text
temperature_difference =
process_temperature - air_temperature
```

## ⚙️ Data Processing

Raw machine observations are transformed into model-ready features through a shared preprocessing module.

```text
Raw machine data
       |
       v
Column normalization
       |
       v
Feature engineering
       |
       v
One-hot encoding of machine type
       |
       v
Model features
```

The final model uses:

```text
air_temperature
process_temperature
rotational_speed
torque
tool_wear
temperature_difference
type_H
type_L
type_M
```

The same preprocessing logic is shared between training and inference to prevent training-serving inconsistencies.

## 🤖 Model Development

A simple always-normal classifier was first used as a baseline.

Because only 3.39% of observations correspond to failures, this baseline reaches approximately **96.6% accuracy while detecting zero failures**.

This demonstrates why precision, recall and F1-score are more relevant than accuracy alone for this problem.

Several approaches were then compared:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Always-normal baseline | 96.60% | 0.00% | 0.00% | 0.00% |
| Logistic Regression | 96.70% | 58.33% | 10.29% | 17.50% |
| Balanced Logistic Regression | 82.55% | 14.25% | 82.35% | 24.30% |
| **Balanced Random Forest** | **98.65%** | **83.61%** | **75.00%** | **79.07%** |

The Random Forest provides the best overall precision/recall trade-off among the tested models.

On the held-out test set it detects:

```text
51 / 68 failures
```

with only:

```text
10 false maintenance alerts
```

## 🌲 Why Random Forest?

Exploratory analysis suggested nonlinear relationships between variables such as torque, rotational speed and tool wear.

Random Forest was therefore tested as an alternative to Logistic Regression because tree-based models can capture nonlinear decision boundaries and interactions between variables.

The selected model uses class weighting to account for the strong class imbalance.

## 🏭 Simulated Machine Monitoring

The trained model is persisted with `joblib` and reused by the monitoring simulator without retraining.

The simulator replays observations from the held-out test dataset:

```text
Machine observation
        |
        v
Data preprocessing
        |
        v
Random Forest
        |
        v
Failure score
        |
        v
Decision threshold
        |
        v
Maintenance alert
```

Example:

```text
-----------------------------
Machine: L54191
Type: L
Torque: 12.2 Nm
Tool wear: 17 min

Real status: FAILURE
Failure score: 96.00%

⚠ MAINTENANCE ALERT
```

The current simulator is a **local sequential replay**, not a production real-time streaming system.

A threshold of `0.60` is currently used experimentally for the simulator. This threshold has not yet been validated as an optimal industrial decision threshold.

## 🏗️ Current Architecture

```text
                 TRAINING

       AI4I historical dataset
                 |
                 v
          Data preprocessing
                 |
                 v
          Feature engineering
                 |
                 v
          Train / Test split
                 |
                 v
           Random Forest
                 |
                 v
        random_forest.pkl


                 INFERENCE

       Test machine observation
                 |
                 v
          Data preprocessing
                 |
                 v
        random_forest.pkl
                 |
                 v
          Failure score
                 |
                 v
        Maintenance alert
```

## 📁 Project Structure

```text
industrial-predictive-maintenance/
|
├── data/
│   └── ai4i2020.csv
|
├── models/
│   └── random_forest.pkl
|
├── src/
│   ├── data_processing.py
│   ├── explore.py
│   ├── simulator.py
│   └── train.py
|
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Running the Project

Create and activate a Python virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate the models:

```bash
python src/train.py
```

Run the machine monitoring simulation:

```bash
python src/simulator.py
```

## 🛠️ Technologies

**Current version**

Python · Pandas · NumPy · Matplotlib · scikit-learn · Joblib · Git

## 🗺️ Roadmap

The current ML prototype is the first stage of a broader Data Engineering, AI and Cloud platform.

Planned extensions include:

```text
Industrial / simulated sensors
            |
            v
          Kafka
            |
            v
   Streaming data pipeline
            |
            v
      PySpark processing
            |
            v
       AWS Data Lake
            |
            v
       ML inference
            |
            v
     API / Monitoring
```

Future work will focus on:

- Kafka-based event streaming
- PySpark data processing
- Docker containerization
- AWS S3 Data Lake integration
- Model serving through an API
- Monitoring and observability
- Model and threshold validation
- Deep Learning / time-series predictive maintenance

## ⚠️ Current Limitations

The AI4I dataset is synthetic and does not represent measurements collected from a real production machine.

The current monitoring component replays historical test observations sequentially and should therefore not be interpreted as a production real-time system.

The model's `predict_proba` output is used as a failure score/probability estimate but has not yet undergone probability calibration.

The decision threshold also requires validation according to real industrial costs associated with false alarms and missed failures.

## 📚 Dataset Attribution

AI4I 2020 Predictive Maintenance Dataset  
UCI Machine Learning Repository

Dataset licensed under **CC BY 4.0**.