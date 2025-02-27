.. trainer documentation master file, created by
   sphinx-quickstart on Wed Feb 26 17:05:43 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentación del Componente 'Inferer'
=====================

El componente 'Inferer' nos ayuda a generar una predicción
sobre el modelo de XGBoost entrenado.


.. toctree::
   :maxdepth: 2
   :caption: Contenido

   modules

Instrucciones
=============

Clonar repositorio
------------------
Primero, es necesario clonar el repositorio:

.. code-block:: bash

    git clone git@github.com:UlisesRG-ITAM/tarea5_docker.git

Ir a carpeta con Dockerfile
------------------
Acceder a la siguiente carpeta:

.. code-block:: bash

    cd container_infer

Construir la imagen
-------------------
Después de clonar el repositorio, construye la imagen:

.. code-block:: bash

    docker build -t container_infer .

Correr el contenedor
--------------------
Finalmente, corre el contenedor con las siguientes instrucciones:

.. code-block:: bash

    docker run \
        -it \
        --rm \
        -v <ubicacion_donde_se_clono_el_repo>/tarea5_docker/container_infer:/usr/app \
        container_infer \
        --input data/inference.csv --model model.joblib --columns data/trained_columns.json --output data/predictions.csv

Ejemplo de ejecución
====================

Construcción de la imagen:

.. image:: _static/build_infer.png
   :alt: Build Infer
   :width: 1000px
   :align: center

Ejecución del contenedor:

.. image:: _static/run_infer.png
   :alt: Run Infer
   :width: 1000px
   :align: center