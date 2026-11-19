# Models

The trained SavedModels behind the seven `meinhardt4ai/translators` serving
images (`detector`, `de-en`, `en-de`, `de-fr`, `fr-de`, `en-fr`, `fr-en`).

## Where they live

They are **not** in this repository. The six translation models are 185 MB each
and their `variables.data-00000-of-00001` file alone is 167 MB, over GitHub's
hard 100 MB limit for a file in plain git.

They live in two independent places instead:

- the published Docker images on Docker Hub, which is what actually runs; and
- `.tar.gz` assets on the [`models-v1`][release] release of this repository,
  which is free and unmetered for a public repo and allows up to 2 GB per asset.

[release]: https://github.com/s-meinhardt/translators/releases/tag/models-v1

The release copy is deliberate redundancy rather than the working copy. The
original training outputs are gone — the path the old build script read from
(`/home/sven/Python/TensorFlow-Training/Text/Translator/<name>/beamsearch/000<n>`)
no longer exists, and `s-meinhardt/tensorflow-training` is empty — so until this
release the images were the *only* copy of six years' worth of training. Pulling
a model out of an image is easy enough, but it only works while the image is
still there.

| Asset | Size | SHA-256 |
|---|---|---|
| `detector.tar.gz` | 1 MB | `6032587fe92a3dc7ecde637d0c43ce9f98eeff557075c78069532a75f8747505` |
| `de-en.tar.gz` | 153 MB | `b56a6a6c8ceef65fb9b1097fcd8062719030659afcb8b509c488f266165dbc27` |
| `en-de.tar.gz` | 153 MB | `3c1504be085260757ca3ba41432267890dd780670b2e4dec1e2961a547cabe09` |
| `de-fr.tar.gz` | 153 MB | `81a047ae22aff4626b37ae86533131723e586678ae835c10bd841ddc0bcacb8f` |
| `fr-de.tar.gz` | 152 MB | `799bb7f32b8308896812984b7af5c34c4ce68312f52dbe6c27330932235c934d` |
| `en-fr.tar.gz` | 152 MB | `9d972406a2df1547626b8d090f5822b773ab8afe991df1173fe59adbae0e2468` |
| `fr-en.tar.gz` | 152 MB | `87f438ece35018662083194c98f31d8b74b81343a62991c84d7174f5f2921299` |

Each archive unpacks to a single `0001/` directory holding `saved_model.pb` and
`variables/` — the layout `tensorflow/serving` expects for one model version.
`SHA256SUMS` on the release covers all seven.

They were recovered from the images with
[`scripts/extract-models-from-images.sh`](../scripts/extract-models-from-images.sh),
which is also how you would recover them again.

[`serving/models.txt`](../serving/models.txt) records which release each image's
model is built from. It exists because one release tag for the whole set does
not survive a retrain: a retrained model goes to a new release holding only
that archive, so a later rebuild of all seven would either revert it (pointing
at the old release) or fail to find the other six (pointing at the new one).
With a pin per model, a rebuild of all seven always takes each model from the
release it is actually live in, and a retrain is a reviewable one-line diff.
`docker-script` reads the pins when no `--release` is given, and
`package-model.sh` updates the relevant line after a successful upload.

The repository's git history contains only older, superseded `de-en` variants —
the `de-en v2` and `de-en v3` directories with two-shard variables, and a
`de-en v4` tree — not the deployed models. Together they account for roughly
407 MB of the repository's history.

## Signatures

`detector` — classifies a 64-token sequence as German, English or French:

| | Name | Type | Shape |
|---|---|---|---|
| input | `embedding_input` | float32 | `[-1, 64]` |
| output | `dense_2` | float32 | `[-1, 3]` |

The six translation models share one signature and are all served under the model
name `translator`, the web app telling them apart by port:

| | Name | Type | Shape |
|---|---|---|---|
| input | `encoder_input` | int32 | `[-1, -1]` |
| output | `sequences` | int32 | `[-1, -1]` |
| output | `probabilities` | float32 | `[-1]` |

Beam search is baked into the SavedModel, so a request returns finished token
sequences and their probabilities; the web app only tokenizes and detokenizes.

## tensorflow/serving version

The deployed images were built on `tensorflow/serving:2.2.0` in August 2020.
`docker-script` and the `serving images` workflow now default to **2.21.0**, the
current release, so a rebuild picks up several years of base-image fixes. The
image tags do not change, so [translator.yml](../translator.yml) needs no edit.

All seven models were checked on 2.21.0 by rebuilding each image, querying it
and running a prediction. All seven load and all seven predict correctly.

### Use real token ids when testing a model

A prediction only means something if the input is a real tokenised sentence.
Arbitrary ids make beam search wander until TensorFlow Serving's own 30-second
REST timeout fires and returns `{"error": "Timed out waiting for notification"}`
— which says nothing about the model.

An earlier revision of this file reported `en-fr` as unable to predict, based on
an input of `[2, 100, 200, 300, 400, 5, 0…]`. Those ids are
`[CLS] ¢ ċ Ι н !` in `flask-app/vocabularies/wmt_en_32k.txt`, and the web app
strips `[CLS]`/`[SEP]` before sending anyway (`translator.py`, `.ids[1:-1]`).
That record was wrong: given a real sentence, `en-fr` answers in about 25
seconds under amd64 emulation, on both `2.2.0` and `2.21.0`:

```
$ curl -d '{"instances":[[6868,7272,16,2414,1751,1874,35]]}' \
    localhost:8501/v1/models/translator:predict          # Good morning , how are you ?
{"predictions":[{"probabilities":0.00318805268,
                 "sequences":[9731,8481,16,3423,5767,17,2114,35,0,0]}]}
                                                        # Bon matin , comment êtes - vous ?
```

The inputs the `serving images` workflow uses are whole-word lookups in the
app's own vocabularies, one per source language:

| Models | Token ids | Sentence |
|---|---|---|
| `en-de`, `en-fr` | `6868,7272,16,2414,1751,1874,35` | Good morning , how are you ? |
| `de-en`, `de-fr` | `19174,6793,16,1956,3364,1922,2794,35` | Guten Morgen , wie geht es dir ? |
| `fr-en`, `fr-de` | `32540,16,3423,15248,17,2114,35` | Bonjour , comment allez - vous ? |
| `detector` | 64 × `0.1` | (neutral embedding input) |

Because the input is real, the prediction is a **gate**: an image that loads its
model but cannot translate is not pushed.

Note that `tensorflow/serving` publishes **amd64 only** — there is no arm64
manifest — which is why these images are built on GitHub's amd64 runners rather
than locally on an Apple Silicon machine.

## Retraining a model

Training happens locally in [`translator_de_en.ipynb`](../translator_de_en.ipynb);
everything after it happens on GitHub, so no local path ends up in the repository
or in an image.

1. Train the model in the notebook. It writes a SavedModel to a relative path of
   its own (`./test/transformer` by default).
2. Package it and upload it to a **new** release:
   ```bash
   scripts/package-model.sh de-en <the-saved-model-dir> --release models-v2
   ```
   Always a new tag. The script refuses to replace an archive that is already
   published, because for six of the seven models the release copy is the only
   one outside the Docker Hub image — overwriting it and then rebuilding that
   image would leave the original model nowhere.
3. Commit the one-line change the script made to
   [`serving/models.txt`](../serving/models.txt) **and get it onto `master`**
   (push it, or merge its PR). The workflow checks out the ref it was
   dispatched on, which is the default branch unless you pass `--ref`, so it
   reads `master`'s copy of the pins — a pin that exists only locally or on a
   branch would rebuild the *previous* archive, and the run would still be
   green. To try a pin before merging, dispatch with
   `--ref <branch> -f push=false`.
4. Rebuild and push that image. Naming the release explicitly is correct for a
   single model whether or not the pin has landed yet:
   ```bash
   gh workflow run "serving images" \
       -f models=de-en -f release=models-v2 -f push=true
   ```
   Or use **Actions → serving images → Run workflow** in the browser. Once the
   pin is on `master`, a rebuild of everything needs no release at all — the
   pins say where each model comes from. Each run reports, in its summary,
   which release every image was built from.

The workflow refuses to push an image whose model does not load, so a bad archive
fails the run rather than the server.

## Rolling back a serving image

Unlike `web_app`, which also gets a `web_app-<sha>` tag, the serving images are
pushed only as `:<model>` — there is no previous tag to repoint to. Rolling back
means rebuilding, which is reliable now that the models are archived:

```bash
gh workflow run "serving images" -f models=de-en -f tf_serving_version=2.2.0 -f push=true
```

That reproduces the image as it was on the old base. To roll back to an earlier
*model*, set that model's line in `serving/models.txt` back to the older release
and its checksum, get that onto `master`, and rerun the workflow — or pass
`-f release=<older tag>` for a single model, which does not wait on `master`.
