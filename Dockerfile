FROM ubuntu

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update

RUN apt-get install -y libgl1-mesa-dev

RUN mkdir /ris

WORKDIR /ris

COPY . ./

RUN pip3 install -r requirements.txt

ENTRYPOINT ["/bin/bash"]