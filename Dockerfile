FROM python:3.12

ADD . /opt/ml_in_app
WORKDIR /opt/ml_in_app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["sh", "-c", "python Download.py && python APPNOVA.py"]