#!/bin/bash
./opt/conda/etc/profile.d/conda.sh
conda activate base
conda install spacy==1.9.0
pip install -r /nae-gc-re-system/python/requirements.txt
python app.py
