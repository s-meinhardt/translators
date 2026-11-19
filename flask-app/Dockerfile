FROM python:3.12-slim

WORKDIR /deploy/

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK's sentence-splitting data and fail the build if that does not work
# (nltk.download only returns False on failure). Since nltk 3.10.3 downloads through an
# HTTP(S) proxy are refused; when building behind a proxy you trust, pass
#   --build-arg NLTK_ALLOW_PROXIED_URLOPEN=1
ARG NLTK_ALLOW_PROXIED_URLOPEN=0
RUN python -c "import sys, nltk; sys.exit(0 if nltk.download('punkt_tab') else 1)"

# smoke test: importing the app runs its startup check, which loads the data
RUN python -c "import translator"

EXPOSE 5000

ENTRYPOINT ["python", "translator.py"]
