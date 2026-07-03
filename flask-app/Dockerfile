FROM python:3.12-slim

WORKDIR /deploy/

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt_tab')"

EXPOSE 5000

ENTRYPOINT ["python", "translator.py"]
