# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import json
import gzip
import pickle
import zipfile
from glob import glob

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def load_datasets(directory):
    """Carga archivos CSV dentro de archivos ZIP desde un directorio dado."""
    datasets = []
    for filepath in glob(f"{directory}/*"):
        with zipfile.ZipFile(filepath, "r") as zf:
            for filename in zf.namelist():
                with zf.open(filename) as f:
                    datasets.append(pd.read_csv(f, sep=",", index_col=0))
    return datasets

def prepare_output_dir(output_dir):
    """Elimina y recrea el directorio de salida si existe."""
    if os.path.exists(output_dir):
        for file in glob(f"{output_dir}/*"):
            os.remove(file)
        os.rmdir(output_dir)
    os.makedirs(output_dir)

def clean_dataset(df):
    """Aplica limpieza de datos eliminando valores no válidos."""
    df = df.copy()
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df = df[df["MARRIAGE"] != 0]
    df = df[df["EDUCATION"] != 0]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: min(x, 4))
    return df.dropna()

def separate_features_target(df):
    """Divide el dataset en características y variable objetivo."""
    return df.drop(columns=["default"]), df["default"]

def build_pipeline():
    """Crea una pipeline para procesamiento y clasificación."""
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
        remainder="passthrough",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])

def configure_estimator(pipeline):
    """Configura la búsqueda de hiperparámetros."""
    param_grid = {
        "classifier__n_estimators": [100, 200, 500],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }
    return GridSearchCV(
        pipeline, param_grid, cv=10, scoring="balanced_accuracy", n_jobs=-1, refit=True, verbose=2
    )

def save_model(filepath, estimator):
    """Guarda el modelo entrenado en un archivo comprimido."""
    prepare_output_dir("files/models/")
    with gzip.open(filepath, "wb") as f:
        pickle.dump(estimator, f)

def compute_metrics(dataset_name, y_true, y_pred):
    """Calcula métricas de evaluación."""
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

def compute_confusion_matrix(dataset_name, y_true, y_pred):
    """Genera la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }

def main():
    """Ejecuta el proceso de carga, entrenamiento y evaluación del modelo."""
    test_df, train_df = [clean_dataset(df) for df in load_datasets("files/input")]  
    x_train, y_train = separate_features_target(train_df)
    x_test, y_test = separate_features_target(test_df)
    pipeline = build_pipeline()
    estimator = configure_estimator(pipeline)
    estimator.fit(x_train, y_train)
    save_model("files/models/model.pkl.gz", estimator)
    
    metrics = [
        compute_metrics("train", y_train, estimator.predict(x_train)),
        compute_metrics("test", y_test, estimator.predict(x_test)),
        compute_confusion_matrix("train", y_train, estimator.predict(x_train)),
        compute_confusion_matrix("test", y_test, estimator.predict(x_test)),
    ]
    os.makedirs("files/output/", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        file.writelines(json.dumps(m) + "\n" for m in metrics)

if __name__ == "__main__":
    main()
