# Dockerfile
FROM python:3.10.2
WORKDIR /usr/app
RUN python -m pip install \
    joblib==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scipy==1.14.0 \
    scikit-learn==1.5.1
COPY prep.py .
COPY src/preprocessing.py .
COPY data/raw.csv .
ENTRYPOINT ["python", "prep.py"]
CMD []