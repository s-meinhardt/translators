# translators/de-en
This folder contains a jupyter notebook to train a German --> English translator based on Google's Transformer architecture. The model will be trained on the WMT19/de_en dataset. 

You can use the vocabularies provided in the 'vocabularies' folder to initialize your tokenizers, thus saving lots of time. Just copy the vocabuly files into the vocabulary folder specified in the notebook.

The flask-app folder contains all necessary files to create a docker image of a flask web server including the translator model. Once the docker image has been build and is running, one can use the web browser or the provided RESTful API to request translations. Visit http://meinhardt.spdns.org:8080/translator to test the app. You can also download a pre-built image from https://hub.docker.com/r/meinhardt4ai/translators using the name 

meinhardt4ai/translators:de_en 
 
