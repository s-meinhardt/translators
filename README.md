# translators

A machine-learning translator for German, English and French, based on Google's Transformer architecture. The models are trained on the WMT19 datasets. A separate detector model recognizes the input language automatically, so the caller only needs to specify the desired output language.

Visit <http://meinhardt.spdns.org:8080/translator> to try the app. Pre-built images are available from <https://hub.docker.com/r/meinhardt4ai/translators>.

## Architecture

The full stack is eight containers, wired together by [translator.yml](translator.yml):

- **web** — a Flask app ([flask-app/](flask-app)) that serves the UI and a small REST API.
- **detector** — a TensorFlow Serving instance that classifies the input text as German, English or French.
- **de-en, en-de, de-fr, fr-de, en-fr, fr-en** — one TensorFlow Serving instance per translation direction.

The web app tokenizes requests with pre-built WordPiece vocabularies, forwards them to the detector and the matching translation service over HTTP, and renders the result.

The seven serving images each carry one trained model baked into it. The models themselves are too big for git; they are archived as release assets and documented in [docs/models.md](docs/models.md).

## Repository layout

| Path | Description |
|---|---|
| [translator.yml](translator.yml) | docker-compose file that runs the full application: the web app, the language detector (port 8600) and the six translation services de-en, en-de, de-fr, fr-de, en-fr, fr-en (ports 8601–8606). |
| [flask-app/](flask-app) | Source and Dockerfile for the web server image (`meinhardt4ai/translators:web_app`). |
| [flask-app/vocabularies/](flask-app/vocabularies) | Pre-built tokenizer vocabularies, used by the web server at runtime. They can also initialize your tokenizers when training, thus saving lots of time: just copy the vocabulary files into the vocabulary folder specified in the notebook. |
| [translator_de_en.ipynb](translator_de_en.ipynb) | Jupyter notebook to train a translator model. |
| [docker-script](docker-script) | Builds (and optionally pushes) one `tensorflow/serving` image from an archived model, tagged `meinhardt4ai/translators:<name>`. |
| [serving/Dockerfile](serving/Dockerfile) | Dockerfile behind `docker-script`: lays one SavedModel into a `tensorflow/serving` base. |
| [serving/models.txt](serving/models.txt) | Which release each serving image's model is built from, one line per model. |
| [scripts/package-model.sh](scripts/package-model.sh) | Packages a freshly trained model and uploads it, ready for the serving-image workflow. |
| [scripts/extract-models-from-images.sh](scripts/extract-models-from-images.sh) | Recovers the models from the published images. |
| [docs/models.md](docs/models.md) | Where the trained models are archived, their signatures, and how to retrain one. |
| [.github/workflows/](.github/workflows) | CI: builds the web image on every change, and rebuilds the serving images on demand. |

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

Training is the only step that happens on your machine. Everything after it runs
on GitHub, so no local path ends up in the repository or in an image.

1. Open [translator_de_en.ipynb](translator_de_en.ipynb) and train a model, optionally seeding the tokenizer with a vocabulary from [flask-app/vocabularies/](flask-app/vocabularies). It writes a SavedModel to a relative path of its own.
2. Package it and upload it to a new release (always a new tag — the script
   refuses to overwrite a published archive):
   ```bash
   scripts/package-model.sh de-en <the-saved-model-dir> --release models-v2
   ```
3. Commit the one-line change it makes to [serving/models.txt](serving/models.txt).
4. Rebuild and push `meinhardt4ai/translators:de-en`:
   ```bash
   gh workflow run "serving images" -f models=de-en -f push=true
   ```

See [docs/models.md](docs/models.md) for the archive layout and the model signatures.

## Building and publishing images

All images are built by GitHub Actions. That is not only convenience:
`tensorflow/serving` is published for amd64 only, so the seven serving images
build natively on a GitHub runner and would need emulation on an Apple Silicon
machine.

### `web app image`

This is the one that runs by itself, and the only one a dependency bump touches —
`flask-app/requirements.txt` is all those change, and `web_app` is the only image
built from it. The seven serving images carry frozen models and are left alone.

It runs on every pull request and push that touches [flask-app/](flask-app),
builds the image, starts it and checks that the app serves its page, then pushes
`meinhardt4ai/translators:web_app` (plus a `web_app-<sha>` tag to roll back to)
on a push to `master`. Pull requests are built and tested but not pushed, so a
dependabot PR is verified before it is merged.

### `serving images`

Manual only (**Actions → serving images → Run workflow**), because the models are
frozen: the reasons to rebuild are moving onto a newer `tensorflow/serving` base
and publishing a retrained model. Inputs:

| Input | Default | |
|---|---|---|
| `models` | `all` | Space-separated subset, e.g. `de-en fr-en`. |
| `tf_serving_version` | `2.21.0` | Base image tag — see [docs/models.md](docs/models.md#tensorflowserving-version). |
| `release` | *(empty)* | Overrides the release for every model; normally left empty so the per-model pins in [serving/models.txt](serving/models.txt) apply. |
| `push` | `false` | Build and verify only, unless set. |

Every image is started and checked for its model reaching `AVAILABLE` before it
is pushed, so a base-image bump that breaks model loading fails the run rather
than the server. A prediction is also attempted and reported, but only as a
warning — see [docs/models.md](docs/models.md#tensorflowserving-version), which
records what was verified on the current base and the one open question about
`en-fr`.

### Required secrets

Both workflows push to Docker Hub and need two repository secrets:

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub account name. |
| `DOCKERHUB_TOKEN` | A Docker Hub access token with **Read & Write** on `meinhardt4ai/translators`. |

Pull-request builds do not log in, so they work without them.

## Tech stack

Flask, Werkzeug, NLTK (sentence splitting), Hugging Face `tokenizers` (WordPiece), TensorFlow Serving, Docker Compose.
