# GC Rec System

### To Run
1) Set up Anaconda
```
conda create env --name nae-gc
conda install spacy==1.9.0
python -m spacy download en_core_web_md
```

2) Install the rest of the dependencies using pip
```
pip install -r requirements.txt
python app.py
```

3) Deploying
At terminal set up google cloud SDK, Docker
```
// Here project ID is get-schooled-7e158
$ cd python
$ docker build -t nae-gc -< Dockerfile
$ gcloud auth configure-docker
$ docker tag nae-gc gcr.io/[PROJECT-ID]/nae-gc
$ docker push gcr.io/[PROJECT-ID]/nae-gc
$ gcloud app deploy nae-gc gcr.io/[PROJECT-ID]/nae-gc
```