#!/usr/bin/env bash
#
# Package a freshly trained model into the archive layout the "serving images"
# workflow expects, and optionally upload it to a release.
#
# This is the handover point between training and building: the notebook writes a
# SavedModel somewhere on your machine, this turns it into <model>.tar.gz, and
# from there everything happens on GitHub - so no local path ever ends up in the
# repository or in an image.
#
# Usage:
#   scripts/package-model.sh <model> <saved-model-dir> [options]
#
#   <model>             detector | de-en | en-de | de-fr | fr-de | en-fr | fr-en
#   <saved-model-dir>   The directory holding saved_model.pb and variables/
#                       (one model version, whatever the notebook called it).
#
# Options:
#   --out DIR       Where to write the archive (default: ./extracted-models,
#                   which is git-ignored).
#   --release TAG   Upload the archive to this release, creating it if needed.
#   -h, --help      Show this help.
#
# Example - retrain de-en, then rebuild and push its image entirely on GitHub:
#
#   scripts/package-model.sh de-en ~/training/de-en/0007 --release models-v2
#   # that updates serving/models.txt; commit it, then:
#   gh workflow run "serving images" -f models=de-en -f push=true

set -euo pipefail

VALID_MODELS="detector de-en en-de de-fr fr-de en-fr fr-en"
out="extracted-models"
release=""

die() { printf 'error: %s\n' "$1" >&2; exit 1; }
usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,2\} \{0,1\}//'; }

model=""
src=""
while [ $# -gt 0 ]; do
	case "$1" in
		-h|--help)  usage; exit 0 ;;
		--out)      out="${2:?--out needs a directory}"; shift 2 ;;
		--release)  release="${2:?--release needs a tag}"; shift 2 ;;
		-*)         die "unknown option: $1" ;;
		*)
			if [ -z "$model" ]; then model="$1"
			elif [ -z "$src" ]; then src="$1"
			else die "unexpected argument: $1"
			fi
			shift ;;
	esac
done

[ -n "$model" ] && [ -n "$src" ] || { usage; exit 2; }

case " $VALID_MODELS " in
	*" $model "*) ;;
	*) die "unknown model '$model'; expected one of: $VALID_MODELS" ;;
esac

[ -f "$src/saved_model.pb" ] \
	|| die "$src does not look like a SavedModel version directory (no saved_model.pb)"

mkdir -p "$out"
archive="$out/$model.tar.gz"

# The archive always contains a single 0001/ directory, which is the layout
# serving/Dockerfile copies into place.
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
cp -R "$src" "$staging/0001"

# COPYFILE_DISABLE keeps macOS from adding ._* AppleDouble entries.
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$archive" -C "$staging" 0001
echo "==> wrote $archive ($(du -h "$archive" | cut -f1))"
shasum -a 256 "$archive"

if [ -n "$release" ]; then
	command -v gh >/dev/null || die "gh is required to upload; install it or upload by hand"
	if gh release view "$release" --repo s-meinhardt/translators >/dev/null 2>&1; then
		# Never replace an archive that is already published. For six of the
		# seven models the release copy is the only one outside the Docker Hub
		# image, and overwriting it - then rebuilding and pushing that image -
		# would leave the original model nowhere. A retrain goes to a new tag,
		# which is what docs/models.md describes.
		if gh release view "$release" --repo s-meinhardt/translators \
			--json assets -q '.assets[].name' | grep -qx "$model.tar.gz"; then
			die "release $release already has $model.tar.gz; use a new release tag"
		fi
		echo "==> uploading to existing release $release"
		gh release upload "$release" "$archive" --repo s-meinhardt/translators
	else
		echo "==> creating release $release"
		gh release create "$release" "$archive" --repo s-meinhardt/translators \
			--title "Trained models ${release#models-}" \
			--notes "Model archives for the meinhardt4ai/translators serving images. See docs/models.md."
	fi
	# Record which release this model is now live in, so a later rebuild of all
	# seven images picks up the new archive for this one and the old archives for
	# the rest. Without this the next base-image rebuild would quietly revert it.
	pins="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/serving/models.txt"
	if [ -f "$pins" ]; then
		tmp="$(mktemp)"
		awk -v m="$model" -v r="$release" \
			'$1 == m { printf "%-11s %s\n", m, r; next } { print }' "$pins" > "$tmp"
		if cmp -s "$pins" "$tmp"; then
			rm -f "$tmp"
			echo "==> serving/models.txt already pins $model to $release"
		else
			mv "$tmp" "$pins"
			echo "==> updated serving/models.txt: $model -> $release"
			echo "    commit that one-line change so the repository records it."
		fi
	else
		echo "==> note: $pins not found; pin $model to $release by hand"
	fi

	echo "==> now rebuild the image on GitHub:"
	echo "    gh workflow run 'serving images' -f models=$model -f push=true"
fi
