FROM python:3.11.9

EXPOSE 8501
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "lawcode.py", "--server.port=8501", "--server.address=0.0.0.0"]
