FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip

CMD ["python", "bot_gui.py"]
