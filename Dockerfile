FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
CMD ["sh", "-c", "gunicorn --preload -w 2 -t 120 -b 0.0.0.0:${PORT:-7860} app:lead_conversion_api"]
