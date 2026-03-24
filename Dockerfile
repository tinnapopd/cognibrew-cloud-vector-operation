FROM python:3.10.18-slim

LABEL description="CogniBrew Cloud Vector Operation"
LABEL maintainer="Tinnapop Duangtha <tinnapopduangtha@gmail.com>"

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY app/ ./app/

RUN chmod +x scripts/prestart.sh

EXPOSE 8000

CMD ["bash", "-c", "./scripts/prestart.sh && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
