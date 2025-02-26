"""
Este módulo contiene las funciones necesarias 
para poder realizar el pre-procesamiento de variables.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import entropy


def identify_variable_types(df: pd.DataFrame):
    """
    Identifica las columnas numéricas y categóricas en un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        logger: Objeto de logging para registrar eventos.

    Returns:
        tuple: Listas de columnas numéricas y categóricas.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols


def impute_missing_values(df: pd.DataFrame, num_cols: list, cat_cols: list):
    """
    Imputa valores nulos: 0 para variables numéricas y 'Without It' para categóricas.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        num_cols (list): Lista de columnas numéricas.
        cat_cols (list): Lista de columnas categóricas.
        logger: Objeto de logging para registrar eventos.

    Returns:
        pd.DataFrame: DataFrame con valores imputados.
    """
    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna("Without It")
    return df


def select_relevant_features(
    df: pd.DataFrame, num_cols: list, cat_cols: list, target: str
):
    """
    Selecciona variables relevantes:
    - Numéricas con correlación mínima de 0.25 con el target.
    - Categóricas con entropía mínima de 1.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        num_cols (list): Lista de columnas numéricas.
        cat_cols (list): Lista de columnas categóricas.
        target (str): Nombre de la variable objetivo.
        logger: Objeto de logging para registrar eventos.

    Returns:
        list: Lista de columnas relevantes.
    """
    relevant_num_cols = [
        col for col in num_cols if abs(df[col].corr(df[target])) >= 0.25
    ]

    relevant_cat_cols = []
    for col in cat_cols:
        ent = entropy(df[col].value_counts(normalize=True), base=2)
        if ent >= 1:
            relevant_cat_cols.append(col)

    return relevant_num_cols, relevant_cat_cols


def transform_features(df: pd.DataFrame, num_cols: list, cat_cols: list):
    """
    Escala variables numéricas y convierte categóricas a One-Hot Encoding.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        num_cols (list): Lista de columnas numéricas.
        cat_cols (list): Lista de columnas categóricas.
        logger: Objeto de logging para registrar eventos.

    Returns:
        pd.DataFrame: DataFrame transformado.
    """
    # Escalado de variables numéricas
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # One-Hot Encoding para variables categóricas
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    encoded_cats = pd.DataFrame(encoder.fit_transform(df[cat_cols]))
    encoded_cats.columns = encoder.get_feature_names_out(cat_cols)

    df_final = pd.concat([df[num_cols], encoded_cats], axis=1)
    return df_final


def preprocess_data(filepath: str, target: str, logger):
    """
    Función principal que carga los datos, realiza la limpieza y transformación.

    Args:
        filepath (str): Ruta del archivo CSV de entrada.
        target (str): Nombre de la variable objetivo.
        logger: Objeto de logging para registrar eventos.

    Returns:
        pd.DataFrame: DataFrame preprocesado listo para el entrenamiento.
    """
    try:
        df = pd.read_csv(filepath)
        logger.info("Datos cargados correctamente")

        num_cols, cat_cols = identify_variable_types(df)
        df = impute_missing_values(df, num_cols, cat_cols)
        relevant_num_cols, relevant_cat_cols = select_relevant_features(
            df, num_cols, cat_cols, target
        )
        df = transform_features(df, relevant_num_cols, relevant_cat_cols)

        logger.info("Preprocesamiento completado exitosamente")
        return df
    except Exception as e:
        logger.error(f"Error en el preprocesamiento: {e}")
        return None
