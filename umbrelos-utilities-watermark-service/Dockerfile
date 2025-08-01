FROM python:3.9-slim

WORKDIR /app

RUN pip install flask pillow

# Instalar dependencias necesarias para procesamiento de imágenes y vídeos
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 \
    && pip install numpy opencv-python \
    && rm -rf /var/lib/apt/lists/*

COPY app /app

EXPOSE 9009

CMD ["python", "server.py"]
