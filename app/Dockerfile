FROM python:3.10

WORKDIR /app

# Installing libGL library as OpenCV requires it with docker
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt .

RUN pip install --upgrade pip  && \ 
pip install --no-cache-dir -r requirements-app.txt

COPY . .

EXPOSE 8501

CMD ["streamlit","run","modern_interface.py", "--server.port=8501","--server.address=0.0.0.0"]
