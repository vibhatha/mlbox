FROM ubuntu:18.04
MAINTAINER MLPerf MLBox Working Group

# Remove all stopped containers: docker rm $(docker ps -a -q)
# Remove containers like none:none: docker rmi $(docker images | grep none | awk '{print $3}')

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            software-properties-common \
            python3-dev \
            unzip \
            libgeos-dev \
            git-all \
            curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

RUN apt-get update && \
    apt install -y build-essential

RUN pip3 install git+https://github.com/matplotlib/basemap.git

COPY hydrodl.py /workspace/hydrodl.py
COPY camels.zip /workspace/data/camels.zip

ENTRYPOINT ["python3", "/workspace/hydrodl.py"]