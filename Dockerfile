FROM python:3.10-slim

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt
RUN apt-get update && apt-get install python3-opencv  -y

WORKDIR /app

COPY ./app /app
WORKDIR /

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT [ "python", "-u", "-m", "streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0" ]