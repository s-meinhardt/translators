#!/usr/bin/env bash
#
# Recover the trained SavedModels from the published tensorflow/serving images.
#
# The training code that produced them is not in this repository and the paths
# the old build script used (/home/sven/Python/TensorFlow-Training/...) are gone,
# so the meinhardt4ai/translators images are the surviving copy. This script
# pulls each image, copies the model out of it and writes one .tar.gz per model,
# ready to be attached to a GitHub release (see docs/models.md).
#
# Usage:
#   scripts/extract-models-from-images.sh [output-dir]
#
# Default output directory: ./extracted-models (git-ignored).
#
# Afterwards, publish them with:
#   gh release create models-v1 <output-dir>/*.tar.gz <output-dir>/SHA256SUMS \
#       --title "Trained models v1" --notes "..."

set -euo pipefail

IMAGE_REPO="meinhardt4ai/translators"
MODELS="detector de-en en-de de-fr fr-de en-fr fr-en"
out="${1:-extracted-models}"

command -v docker >/dev/null || { echo "error: docker is not installed" >&2; exit 1; }

mkdir -p "$out"

for model in $MODELS; do
	# The detector image serves /models/detector/0001; the translation images
	# serve /models/translator/0001.
	if [ "$model" = detector ]; then
		src=/models/detector/0001
	else
		src=/models/translator/0001
	fi

	echo "==> $IMAGE_REPO:$model"
	docker image inspect "$IMAGE_REPO:$model" >/dev/null 2>&1 \
		|| docker pull --platform linux/amd64 "$IMAGE_REPO:$model"

	# A container only has to exist to be copied out of, not to run - which
	# matters on hosts that cannot execute these amd64-only images.
	cid="$(docker create --platform linux/amd64 "$IMAGE_REPO:$model")"
	tmp="$(mktemp -d)"
	docker cp "$cid:$src" "$tmp/0001"
	docker rm "$cid" >/dev/null

	# COPYFILE_DISABLE keeps macOS from adding ._* AppleDouble entries.
	COPYFILE_DISABLE=1 tar --no-xattrs -czf "$out/$model.tar.gz" -C "$tmp" 0001
	rm -rf "$tmp"
	echo "    $out/$model.tar.gz ($(du -h "$out/$model.tar.gz" | cut -f1))"
done

( cd "$out" && shasum -a 256 *.tar.gz > SHA256SUMS )
echo "==> wrote $out/SHA256SUMS"
