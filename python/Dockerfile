FROM python:3.6-jessie

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y  \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git

RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		g++ \
		gfortran \
		git \
		liblcms2-dev \
		libopenblas-dev \
		liblapack-dev \
		libssl-dev \
		libzmq3-dev \
		pkg-config \
		software-properties-common \
		libjasper-dev \
		libgdal-dev \
		libdc1394-22-dev \
		libswscale-dev \
		libtheora-dev \
		libvorbis-dev \
		yasm \
		libeigen3-dev \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*  

RUN apt-get update && \
	DEBIAN_FRONTEND=noninteractive apt-get -yq install sqlite3 && \
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV NAE_HOME /nae/python
RUN pip install spacy==1.9.0
RUN python -m spacy download en_core_web_md

RUN git clone https://github.com/vvmurthy/nae-gc-rec-system.git nae/
WORKDIR /nae
RUN cd /nae
WORKDIR /nae/python
RUN cd /nae/python
RUN pip install -r requirements.txt
ENV PATH $NAE_HOME:$PATH
RUN chmod +x /nae/python/run.sh

EXPOSE 5000
CMD /nae/python/run.sh