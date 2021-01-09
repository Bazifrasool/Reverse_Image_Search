FROM ubuntu

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt

RUN mkdir /ris

WORKDIR /ris

COPY . ./

ENTRYPOINT ["/bin/bash"]