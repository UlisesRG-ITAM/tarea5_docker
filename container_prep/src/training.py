"""
Este módulo contiene las funciones necesarias 
para realizar el entrenamiento de un modelo basado en XGBoost.
"""

import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_xgboost(df: pd.DataFrame, target: str, logger):
    """
    Entrena un modelo XGBoost y encuentra los mejores hiperparámetros mediante GridSearchCV.

    Args:
        df (pd.DataFrame): DataFrame de entrada con datos preprocesados.
        target (str): Nombre de la variable objetivo.
        logger: Objeto de logging para registrar eventos.

    Returns:
        xgb.XGBRegressor: Modelo entrenado.
    """
    try:
        x = df.drop(columns=[target])
        y = df[target]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
        }

        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring="neg_mean_absolute_error", verbose=1
        )
        grid_search.fit(x_train, y_train)

        y_pred = grid_search.best_estimator_.predict(x_test)

        logger.info("Entrenamiento completado. " +
                    f"MAE: {mean_absolute_error(y_test, y_pred)}, "+
                    f"MSE: {mean_squared_error(y_test, y_pred)}")

        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        return None


def save_model(model, filepath: str, logger):
    """
    Guarda el modelo entrenado en un archivo.

    Args:
        model: Modelo XGBoost entrenado.
        filepath (str): Ruta donde se guardará el modelo.
        logger: Objeto de logging para registrar eventos.
    """
    try:
        joblib.dump(model, filepath)
        logger.info(f"Modelo guardado en {filepath}")
    except Exception as e:
        logger.error(f"Error al guardar el modelo: {e}")
