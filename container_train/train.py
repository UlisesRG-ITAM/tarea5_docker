"""
Script que ejecuta el entrenamiento del modelo.
"""
import logging
import argparse
import json
import pandas as pd
from src.training import train_xgboost, save_model


def main():
    """
    Script para ejecutar el entrenamiento del modelo.
    """
    # Configuración de logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Entrenamiento del modelo XGBoost")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ruta del archivo de entrada (prep.csv)",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        required=True,
        help="Ruta donde se guardará el modelo entrenado",
    )
    parser.add_argument(
        "--output_columns",
        type=str,
        required=True,
        help="Ruta donde se guardarán las columnas de entrenamiento",
    )
    parser.add_argument(
        "--target", type=str, required=True, help="Nombre de la variable objetivo"
    )
    args = parser.parse_args()

    logger.info("Iniciando entrenamiento del modelo...")
    df = pd.read_csv(args.input)

    model = train_xgboost(df, args.target, logger)

    if model is not None:
        save_model(model, args.output_model, logger)

        # Guardar las columnas utilizadas en el entrenamiento
        trained_columns = df.drop(columns=[args.target]).columns.tolist()
        with open(args.output_columns, "w") as f:
            json.dump(trained_columns, f)

        logger.info("Modelo guardado en %s", args.output_model)
        logger.info("Columnas de entrenamiento guardadas en %s", args.output_columns)
    else:
        logger.error("Error en el entrenamiento del modelo.")


if __name__ == "__main__":
    main()
