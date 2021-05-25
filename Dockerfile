FROM tensorflow/tensorflow:2.2.0

COPY . /image-embedding-api

WORKDIR /image-embedding-api

ADD . /image-embedding-api

RUN python3 -m pip install -r requirements.txt

EXPOSE 5004

CMD ["python3", "main.py"]