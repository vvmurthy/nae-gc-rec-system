FROM debian:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git    

RUN apt-get -y install software-properties-common

RUN ls -a
RUN git clone https://github.com/vvmurthy/nae-gc-rec-system.git nae/
WORKDIR /nae
RUN cd /nae
WORKDIR /nae/python
RUN cd /nae/python
RUN . /opt/conda/etc/profile.d/conda.sh

RUN apt-get update && \
	DEBIAN_FRONTEND=noninteractive apt-get -yq install sqlite3 && \
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE 5000

ENV NAE_HOME /nae/python
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_md
ENV PATH $NAE_HOME:$PATH
RUN chmod +x /nae/python/run.sh
CMD /nae/python/run.sh