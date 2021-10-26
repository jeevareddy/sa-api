FROM python:3.9.1
ADD . /sa-api
WORKDIR /sa-api
RUN pip install -r requirements.txt
EXPOSE 5000
CMD [ "python", "application.py" ]