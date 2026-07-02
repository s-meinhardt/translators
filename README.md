# translators

A machine-learning translator for German, English and French, based on Google's Transformer architecture. The models are trained on the WMT19 datasets. A separate detector model recognizes the input language automatically.

Visit <http://meinhardt.spdns.org:8080/translator> to test the app. Pre-built images are available from <https://hub.docker.com/r/meinhardt4ai/translators>.

## Repository layout

- `translator_de_en.ipynb` — jupyter notebook to train a translator model.
- `vocabularies/` — pre-built tokenizer vocabularies. Use them to initialize your tokenizers, thus saving lots of time: just copy the vocabulary files into the vocabulary folder specified in the notebook.
- `flask-app/` — all necessary files to build the docker image of the flask web server (`meinhardt4ai/translators:web_app`). Once running, translations can be requested via web browser or the RESTful API.
- `docker-script` — packages a trained model into a `tensorflow/serving`-based docker image tagged `meinhardt4ai/translators:<name>`.
- `translator.yml` — docker-compose file that runs the full application: the web app, the language detector (port 8600) and the six translation services de-en, en-de, de-fr, fr-de, en-fr, fr-en (ports 8601–8606).
