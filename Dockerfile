FROM python:3.7.9

LABEL Author=Stars

WORKDIR /home/project

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY . .

RUN python -m pip install --upgrade pip && pip install -r requirements.txt