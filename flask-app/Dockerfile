FROM python:3.6-slim

WORKDIR /deploy/

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt')"

EXPOSE 5000

ENTRYPOINT ["python", "translator_de_en.py"]
