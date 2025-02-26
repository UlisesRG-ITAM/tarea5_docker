"""
Este módulo contiene las funciones necesarias 
para realizar el la predicción dado un conjunto nuevo de datos.
"""
import pandas as pd
import joblib


def load_model(filepath: str, logger):
    """
    Carga un modelo entrenado desde un archivo.

    Args:
        filepath (str): Ruta del archivo del modelo.
        logger: Objeto de logging para registrar eventos.

    Returns:
        Modelo cargado o None si hay un error.
    """
    try:
        model = joblib.load(filepath)
        logger.info("Modelo cargado exitosamente.")
        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        return None


def prepare_inference_data(df: pd.DataFrame, trained_columns: list, logger):
    """
    Prepara los datos de inferencia asegurando que las columnas coincidan con el modelo entrenado.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        trained_columns (list): Lista de columnas usadas en el entrenamiento.
        logger: Objeto de logging para registrar eventos.

    Returns:
        pd.DataFrame: DataFrame listo para inferencia.
    """
    try:
        missing_cols = set(trained_columns) - set(df.columns)
        if missing_cols:
            for col in missing_cols:
                df[col] = 0  # Agrega columnas faltantes con valores por defecto

        extra_cols = set(df.columns) - set(trained_columns)
        if extra_cols:
            df = df.drop(columns=extra_cols)  # Elimina columnas no utilizadas

        df = df[trained_columns]  # Asegura el orden correcto
        logger.info("Datos de inferencia preparados correctamente.")
        return df
    except Exception as e:
        logger.error(f"Error en la preparación de los datos de inferencia: {e}")
        return None


def make_predictions(model, df: pd.DataFrame, logger):
    """
    Genera predicciones con el modelo cargado.

    Args:
        model: Modelo entrenado.
        df (pd.DataFrame): Datos preprocesados para inferencia.
        logger: Objeto de logging para registrar eventos.

    Returns:
        pd.Series: Predicciones del modelo.
    """
    try:
        predictions = model.predict(df)
        logger.info("Predicciones generadas exitosamente.")
        return pd.Series(predictions)
    except Exception as e:
        logger.error(f"Error al generar predicciones: {e}")
        return None
