# Dockerfile
FROM python:3.10.2
WORKDIR /usr/app
RUN python -m pip install \
    joblib==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    xgboost==2.1.1 \
    scikit-learn==1.5.1
COPY train.py .
COPY src/training.py .
COPY data/prep.csv .
ENTRYPOINT ["python", "train.py"]
CMD []