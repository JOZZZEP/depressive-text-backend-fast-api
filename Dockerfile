FROM python:3.9

RUN pip install --upgrade pip

WORKDIR /DepressiveText

COPY ./requirements.txt /DepressiveText/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /DepressiveText/requirements.txt

COPY ./app /DepressiveText/app
COPY ./model /DepressiveText/model
COPY ./nltk /DepressiveText/nltk
COPY ./tokenizer /DepressiveText/tokenizer

ENV PYTHONPATH="${PYTHONPATH}:/DepressiveText"

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
