# Packages one trained SavedModel into a tensorflow/serving image.
#
# The build context must contain a single model version directory named 0001:
#
#   <context>/0001/saved_model.pb
#   <context>/0001/variables/...
#
# Build it with scripts/build-serving-image.sh rather than by hand; that script
# unpacks the model archive into a context of this shape.

# Overridden by docker-script; see docs/models.md for the verified versions.
ARG TF_SERVING_VERSION=2.21.0
FROM tensorflow/serving:${TF_SERVING_VERSION}

# The six translation directions all serve a model called "translator" (the web
# app hard-codes that name and distinguishes directions by port); the language
# detector serves one called "detector".
ARG MODEL_NAME=translator

COPY 0001 /models/${MODEL_NAME}/0001

# tensorflow/serving reads MODEL_NAME at startup; MODEL_BASE_PATH=/models comes
# from the base image.
ENV MODEL_NAME=${MODEL_NAME}
