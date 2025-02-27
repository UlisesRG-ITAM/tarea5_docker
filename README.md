
# Métodos de Gran Escala: Tarea 5 (tarea5_docker)
## Alumno: Ulises Reyes García
### C.U. 152113

El presente respoitorio contiene 3 carpetas, dentro de las cuales se encuentra el código necesario para crear la imagen y contenedor en Docker que ejecute los siguientes procesos:

1. **container_prep**: Tomar el archivo "data/raw.csv" para aplicar procesos de pre-procesamiento de datos que permitan crear un archivo "data/prep.py" listo para entrenar un modelo de regresión de ML.
2. **container_train**: Tomar el archivo "data/prep.py" para entrenar y validar un modelo de regresión bajo XGBoost.
3. **container_infer**: Tomar el archivo "data/inference.csv" para ejecutar la predicción sobre el modelo de ML entrenado "model.joblib".

============
# Documentación

La carpeta (componente) **container_train** contiene documentación en Sphinx.

Para leer el documento:

container_train/docs/build/html/index.html