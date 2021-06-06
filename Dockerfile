FROM python:3.7

RUN ls

COPY . /usr/share/reverseimagesearch

WORKDIR /usr/share/reverseimagesearch

RUN pip install -r requirements.txt

RUN pip install tensorflow --upgrade --force-reinstall 

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 7000


CMD ["python","app.py"]