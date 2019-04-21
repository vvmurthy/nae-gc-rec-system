FROM python:3.6-jessie

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		g++ \
		git \
		liblcms2-dev \
		libopenblas-dev \
		libssl-dev \
		software-properties-common \
		libeigen3-dev \
        sqlite3 \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*  

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

EXPOSE 8080 
CMD /nae/python/run.sh