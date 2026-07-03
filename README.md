# translators

A machine-learning translator for German, English and French, based on Google's Transformer architecture. The models are trained on the WMT19 datasets. A separate detector model recognizes the input language automatically, so the caller only needs to specify the desired output language.

Visit <http://meinhardt.spdns.org:8080/translator> to try the app. Pre-built images are available from <https://hub.docker.com/r/meinhardt4ai/translators>.

## Architecture

The full stack is eight containers, wired together by [translator.yml](translator.yml):

- **web** — a Flask app ([flask-app/](flask-app)) that serves the UI and a small REST API.
- **detector** — a TensorFlow Serving instance that classifies the input text as German, English or French.
- **de-en, en-de, de-fr, fr-de, en-fr, fr-en** — one TensorFlow Serving instance per translation direction.

The web app tokenizes requests with pre-built WordPiece vocabularies, forwards them to the detector and the matching translation service over HTTP, and renders the result.

## Repository layout

| Path | Description |
|---|---|
| [translator.yml](translator.yml) | docker-compose file that runs the full application: the web app, the language detector (port 8600) and the six translation services de-en, en-de, de-fr, fr-de, en-fr, fr-en (ports 8601–8606). |
| [flask-app/](flask-app) | Source and Dockerfile for the web server image (`meinhardt4ai/translators:web_app`). |
| [flask-app/vocabularies/](flask-app/vocabularies) | Pre-built tokenizer vocabularies, used by the web server at runtime. They can also initialize your tokenizers when training, thus saving lots of time: just copy the vocabulary files into the vocabulary folder specified in the notebook. |
| [translator_de_en.ipynb](translator_de_en.ipynb) | Jupyter notebook to train a translator model. |
| [docker-script](docker-script) | Packages a trained model into a `tensorflow/serving`-based docker image tagged `meinhardt4ai/translators:<name>`. |

## Running the app

```bash
docker compose -f translator.yml up -d
```

The UI is then available at `http://localhost:5000/`. `web` uses `network_mode: host` (it talks to the other seven services on `localhost`), so all eight containers must run on the same machine.

## API

The web app exposes a minimal REST-ish API alongside the UI:

```bash
curl -X POST http://localhost:5000/English/Guten%20Tag
```

`English`, `German` and `French` are valid output languages; the input language is detected automatically. The response is the rendered HTML page — there is no separate JSON endpoint.

## Training a new model

1. Open [translator_de_en.ipynb](translator_de_en.ipynb) and train a model, optionally seeding the tokenizer with a vocabulary from [flask-app/vocabularies/](flask-app/vocabularies).
2. Package the trained model into a serving image:
   ```bash
   ./docker-script <name> <model_version>
   ```
   This produces `meinhardt4ai/translators:<name>`, ready to be referenced in [translator.yml](translator.yml).

## Tech stack

Flask, Werkzeug, NLTK (sentence splitting), Hugging Face `tokenizers` (WordPiece), TensorFlow Serving, Docker Compose.
