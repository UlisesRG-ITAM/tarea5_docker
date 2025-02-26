"""
Script que ejecuta la preparación de los datos
"""
import logging
import argparse
from src.preprocessing import preprocess_data


def main():
    """
    Script para ejecutar el preprocesamiento de datos.
    """
    # Configuración de logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Preprocesamiento de datos")
    parser.add_argument(
        "--input", type=str, required=True, help="Ruta del archivo de entrada (raw.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Ruta del archivo de salida (prep.csv)",
    )
    parser.add_argument(
        "--target", type=str, required=True, help="Nombre de la variable objetivo"
    )
    args = parser.parse_args()

    logger.info("Iniciando preprocesamiento de datos...")
    df_preprocessed = preprocess_data(args.input, args.target, logger)

    if df_preprocessed is not None:
        df_preprocessed.to_csv(args.output, index=False)
        logger.info("Datos preprocesados guardados en %s", args.output)
    else:
        logger.error("Error en el preprocesamiento de datos.")


if __name__ == "__main__":
    main()
