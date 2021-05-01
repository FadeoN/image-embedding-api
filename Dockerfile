FROM tensorflow/tensorflow:2.2.0

COPY requirements.txt /

RUN python3 -m pip install -r requirements.txt

COPY . /image-embedding-api

WORKDIR /image-embedding-api

ADD . /image-embedding-api

ENTRYPOINT [ "run-python"]

EXPOSE 5004

CMD [ "main.py"]