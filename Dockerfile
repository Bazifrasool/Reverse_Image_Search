FROM python:3.7-slim

ENV GENEMBED=TRUE

RUN ls

COPY . /usr/share/reverseimagesearch

WORKDIR /usr/share/reverseimagesearch

RUN pip install -r requirements.txt && pip install tensorflow --upgrade --force-reinstall  &&apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 7000


CMD ["python","app.py"]