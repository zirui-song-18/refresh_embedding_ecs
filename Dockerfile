FROM python:3.12-slim

WORKDIR /app

COPY ecs_tasks/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ecs_tasks/ .

ENTRYPOINT ["python", "entrypoint.py"]
