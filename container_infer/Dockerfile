# Dockerfile
FROM python:3.10.2
WORKDIR /usr/app
RUN python -m pip install \
    joblib==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scipy==1.14.0 \
    xgboost==2.1.1 \
    scikit-learn==1.5.1
COPY inference.py .
COPY model.joblib .
COPY src/inferring.py .
COPY src/preprocessing.py .
COPY data/inference.csv .
COPY data/trained_columns.json .
ENTRYPOINT ["python", "inference.py"]
CMD []