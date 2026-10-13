FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /service
COPY app/model-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY nlp ./nlp
COPY app/model-service/app ./app
COPY app/model-service/healthcheck.py ./healthcheck.py
CMD ["python", "-m", "app.main"]
