FROM python

WORKDIR /app

COPY requirements-app.txt .

RUN pip install --no-cache-dir -r requirements-app.txt

COPY . .

EXPOSE 8501

CMD ["streamlit","run","modern_interface.py", "--server.port=8501","--server.address=0.0.0.0"]
