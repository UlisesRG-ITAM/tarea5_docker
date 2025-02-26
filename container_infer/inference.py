"""
Script que ejecuta la inferencia del modelo.
"""
import logging
import argparse
import json
import pandas as pd
from src.inferring import load_model, prepare_inference_data, make_predictions
from src.preprocessing import (
    identify_variable_types,
    impute_missing_values,
    transform_features,
)


def main():
    """
    Script para ejecutar la inferencia con el modelo entrenado.
    """
    # Configuración de logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Inferencia con el modelo XGBoost")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ruta del archivo de entrada (inference.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta del modelo entrenado (model.joblib)",
    )
    parser.add_argument(
        "--columns",
        type=str,
        required=True,
        help="Ruta del archivo con columnas de entrenamiento (trained_columns.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Ruta del archivo de salida con predicciones (predictions.csv)",
    )
    args = parser.parse_args()

    logger.info("Iniciando inferencia...")
    df = pd.read_csv(args.input)

    # Cargar modelo
    model = load_model(args.model, logger)
    if model is None:
        logger.error("No se pudo cargar el modelo. Terminando ejecución.")
        return

    # Cargar las columnas utilizadas en el entrenamiento
    with open(args.columns, "r") as f:
        trained_columns = json.load(f)

    # Preprocesar los datos de inferencia
    num_cols, cat_cols = identify_variable_types(df)
    df = impute_missing_values(df, num_cols, cat_cols)
    df = transform_features(df, num_cols, cat_cols)
    df = prepare_inference_data(df, trained_columns, logger)

    if df is None:
        logger.error(
            "Error en la preparación de datos para inferencia. Terminando ejecución."
        )
        return

    # Generar predicciones
    predictions = make_predictions(model, df, logger)

    if predictions is not None:
        predictions.to_csv(args.output, index=False, header=["Predictions"])
        logger.info("Predicciones guardadas en %s", args.output)
    else:
        logger.error("Error en la generación de predicciones.")


if __name__ == "__main__":
    main()
