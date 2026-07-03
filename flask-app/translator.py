from flask import Flask, url_for, request, redirect, render_template, jsonify
from markupsafe import escape

import os
import re
import requests
from math import log 

from nltk import sent_tokenize
#from tokenizers import Tokenizer as HFTokenizer
from tokenizers import BertWordPieceTokenizer



# some parameters
INP_SEQ_LENGTH = 40
DET_SEQ_LENGTH = 64
VOCAB_PATH = 'vocabularies'
LANGUAGES = ['German', 'English', 'French']
MODEL = {}
PORT = {}
HOST = {}
Tokenizer = {}




MODEL['Detector'] = 'detector'
HOST['Detector'] = 'http://localhost'   #  tensorflow serving requires this host
PORT['Detector'] = '8600'


MODEL[('German', 'English')] = 'translator'
HOST[('German', 'English')] = 'http://localhost'   #  tensorflow serving requires this host
PORT[('German', 'English')] = '8601'
#Tokenizer[('German', 'English')] = HFTokenizer.from_file(os.path.join(VOCAB_PATH, 'wmt19_de_32.json'))

MODEL[('English', 'German')] = 'translator'
HOST[('English', 'German')] = 'http://localhost'   #  tensorflow serving requires this host
PORT[('English', 'German')] = '8602'
#Tokenizer[('English', 'German')] = HFTokenizer.from_file(os.path.join(VOCAB_PATH, 'wmt19_en_32.json'))

MODEL[('German', 'French')] = 'translator'
HOST[('German', 'French')] = 'http://localhost'   #  tensorflow serving requires this host
PORT[('German', 'French')] = '8603'
#Tokenizer[('German', 'French')] = HFTokenizer.from_file(os.path.join(VOCAB_PATH, 'wmt19_de_32.json'))

MODEL[('French', 'German')] = 'translator'
HOST[('French', 'German')] = 'http://localhost'   #  tensorflow serving requires this host
PORT[('French', 'German')] = '8604'
#Tokenizer[('French', 'German')] = HFTokenizer.from_file(os.path.join(VOCAB_PATH, 'wmt19_en_32.json'))

MODEL[('English', 'French')] = 'translator'
HOST[('English', 'French')] = 'http://localhost'   #  tensorflow serving requires this host
PORT[('English', 'French')] = '8605'
#Tokenizer[('English', 'French')] = HFTokenizer.from_file(os.path.join(VOCAB_PATH, 'wmt19_en_32.json'))

MODEL[('French', 'English')] = 'translator'
HOST[('French', 'English')] = 'http://localhost'   #  tensorflow serving requires this host
PORT[('French', 'English')] = '8606'
#Tokenizer[('French', 'English')] = HFTokenizer.from_file(os.path.join(VOCAB_PATH, 'wmt_de_32.json'))


Tokenizer['Detector'] = BertWordPieceTokenizer(vocab = os.path.join(VOCAB_PATH, 'wmt_32k.txt'), lowercase = False)
Tokenizer['German'] = BertWordPieceTokenizer(vocab = os.path.join(VOCAB_PATH, 'wmt_de_32k.txt'), lowercase = False)
Tokenizer['English'] = BertWordPieceTokenizer(vocab = os.path.join(VOCAB_PATH, 'wmt_en_32k.txt'), lowercase = False)
Tokenizer['French'] = BertWordPieceTokenizer(vocab = os.path.join(VOCAB_PATH, 'wmt_fr_32k.txt'), lowercase = False)







def detect_language(text):

	# tokenizing the input sentences
	sequences = tokenize(text, Tokenizer['Detector'], DET_SEQ_LENGTH)
	

	# correcting the length as the detector requires exactly DET_SEQ_LENGTH
	delta = DET_SEQ_LENGTH - len(sequences[0])
	sequences = [sequence+delta*[0] for sequence in sequences]


	# sending a detection request to a TensorFlow Server instance
	model = MODEL['Detector']
	host = HOST['Detector']
	port = PORT['Detector']
	server_url = f'{host}:{port}/v1/models/{model}:predict'
	headers = {"content-type": "application/json"}
	data = {"signature_name": "serving_default",
			"instances": sequences}
	response = requests.post(server_url, json = data, headers = headers)
	response.raise_for_status()
	results = response.json()['predictions'] 


	# computing the log probabilities for all languages (summing over all sentences) 
	log_probabilities = len(LANGUAGES)*[0] 
	for result in results:
		for i, probability in enumerate(result):
			log_probabilities[i] += log(probability + 1e-15)

	
	# a pure Python implementation of 'argmax'
	index = max(range(len(log_probabilities)), key = lambda i: log_probabilities[i])

	return LANGUAGES[index]





def tokenize(text, tokenizer, max_seq_length):
	
	# spliting the text into sentences
	sentences = sent_tokenize(text, language = 'german')
	

	# tokenizing the sentences and computing the max sequence length
	tmp_sequences = [tokenizer.encode(sentence).ids[1:-1] for sentence in sentences]    
	max_length = min(max([len(sequence) for sequence in tmp_sequences]), max_seq_length)  
  

	# breaking too long sequences and padding with zeros
	sequences = []
	for sequence in tmp_sequences:
		while len(sequence) > 0:
			start_sequence = sequence[: max_seq_length]
			sequences.append(start_sequence + (max_length-len(start_sequence))*[0])
			sequence = sequence[max_seq_length : ]

	return sequences




def clean_translation_en(text, translation):
	l = re.findall(pattern=r'(\d+), (\d+)', string=translation)

	for (d1,d2) in l:
		if d1+','+d2 in text:
			translation = re.sub(pattern=d1+', '+d2, repl=d1+'.'+d2, string=translation)
			
	for (d1,d2) in l:
		if d1+'.'+d2 in text:
			translation = re.sub(pattern=d1+', '+d2, repl=d1+','+d2, string=translation)
			
	l = re.findall(pattern=r'(\d+)\. (\d+)', string=translation)

	for (d1,d2) in l:
		if d1+','+d2 in text:
			translation = re.sub(pattern=d1+'. '+d2, repl=d1+'.'+d2, string=translation)
			
	for (d1,d2) in l:
		if d1+', '+d2 in text:
			translation = re.sub(pattern=d1+'. '+d2, repl=d1+', '+d2, string=translation)
			
	for (d1,d2) in l:
		if d1+'.'+d2 in text:
			translation = re.sub(pattern=d1+'. '+d2, repl=d1+','+d2, string=translation)

	translation = re.sub(pattern=r' ’ s ', repl='’s ', string=translation)
	translation = re.sub(pattern=r' :', repl=':', string=translation)
	translation = re.sub(pattern=r' ;', repl=';', string=translation)
	

	return translation






def clean_translation_de(text, translation):
	l = re.findall(pattern=r'(\d+)\, (\d+)', string=translation)

	for (d1,d2) in l:
		if d1+','+d2 in text:
			translation = re.sub(pattern=d1+', '+d2, repl=d1+'.'+d2, string=translation)
			
	for (d1,d2) in l:
		if d1+'.'+d2 in text:
			translation = re.sub(pattern=d1+', '+d2, repl=d1+','+d2, string=translation)
			
	l = re.findall(pattern=r'(\d+)\. (\d+)', string=translation)

	for (d1,d2) in l:
		if d1+','+d2 in text:
			translation = re.sub(pattern=d1+'. '+d2, repl=d1+'.'+d2, string=translation)
			
	for (d1,d2) in l:
		if d1+', '+d2 in text:
			translation = re.sub(pattern=d1+'. '+d2, repl=d1+', '+d2, string=translation)
			
	for (d1,d2) in l:
		if d1+'.'+d2 in text:
			translation = re.sub(pattern=d1+'. '+d2, repl=d1+','+d2, string=translation)

	translation = re.sub(pattern=r' ’ s ', repl='’s ', string=translation)
	translation = re.sub(pattern=r' :', repl=':', string=translation)
	translation = re.sub(pattern=r' ;', repl=';', string=translation)
	

	return translation




def clean_translation_fr(text, translation):
	l = re.findall(pattern=r'(\d+)\, (\d+)', string=translation)

	for (d1,d2) in l:
		if d1+','+d2 in text:
			translation = re.sub(pattern=d1+', '+d2, repl=d1+'.'+d2, string=translation)
			
	for (d1,d2) in l:
		if d1+'.'+d2 in text:
			translation = re.sub(pattern=d1+', '+d2, repl=d1+','+d2, string=translation)
			
	l = re.findall(pattern=r'(\d+)\. (\d+)', string=translation)

	for (d1,d2) in l:
		if d1+','+d2 in text:
			translation = re.sub(pattern=d1+'. '+d2, repl=d1+'.'+d2, string=translation)
			
	for (d1,d2) in l:
		if d1+', '+d2 in text:
			translation = re.sub(pattern=d1+'. '+d2, repl=d1+', '+d2, string=translation)
			
	for (d1,d2) in l:
		if d1+'.'+d2 in text:
			translation = re.sub(pattern=d1+'. '+d2, repl=d1+','+d2, string=translation)

	translation = re.sub(pattern=r' ’ s ', repl='’s ', string=translation)
	translation = re.sub(pattern=r' :', repl=':', string=translation)
	translation = re.sub(pattern=r' ;', repl=';', string=translation)
	

	return translation




def translate(text, input_language, output_language):
	# do nothing if the input and output language are the same
	if input_language == output_language:
		return text, 100


	# choosing tokenizers and server port according to the input and output language
	model = MODEL[(input_language, output_language)] 
	host = HOST[(input_language, output_language)]
	port = PORT[(input_language, output_language)]


	# sending a translation request to a TensorFlow Server instance
	server_url = f'{host}:{port}/v1/models/{model}:predict'
	headers = {"content-type": "application/json"}
	data = {"signature_name": "serving_default",
			"instances": tokenize(text, Tokenizer[input_language], INP_SEQ_LENGTH)}

	response = requests.post(server_url, json = data, headers = headers)
	response.raise_for_status()   
	
 
	# detokenizing the sequences and computing the probability and the number of all tokens
	# Note: TensorFlow Server returns batches of dictionaries and not dictionary of batches
	translation = ''
	probability = 1   
	n_tokens = 0
	for output in response.json()['predictions']: 
		translation += Tokenizer[output_language].decode(output['sequences']) + ' '
		probability *= output['probabilities']
		n_tokens += len(output['sequences'])


	# cleaning the translation
	if output_language == 'German':
		translation = clean_translation_en(text, translation)
	elif output_language == 'English':
		translation = clean_translation_de(text, translation)
	elif output_language == 'French':
		translation = clean_translation_fr(text, translation)


	# computing the confidence as the (geometric) average of the token probabilities
	confidence = probability**(1. / n_tokens)    
	confidence = int(confidence*100)	# normalizing the confidence


	return translation, confidence





app = Flask(__name__)



@app.route("/")
def home():
	return render_template("translator.html", translation = '', confidence = 0, text = '', input_language = '', output_language = '')



@app.route("/<output_language>/<text>", methods = ["POST"])
def translator(output_language, text):
	try:
		input_language = detect_language(text)
		translation, confidence= translate(text, input_language, output_language)
	except requests.exceptions.RequestException:
		error = f"Sorry, the {output_language} translation service is currently unavailable. Please try again later."
		return render_template("translator.html", translation = '', confidence = 0, text = text, input_language = '', output_language = output_language, error = error)
	return render_template("translator.html", translation = translation, confidence = confidence, text = text, input_language = input_language, output_language = output_language)



@app.route("/en", methods = ["POST"])
def En_translator():
	return redirect(url_for("translator", output_language='English', text=request.form['text']), code=307)
	


@app.route("/de", methods = ["POST"])
def De_translator():
	return redirect(url_for("translator", output_language='German', text=request.form['text']), code=307)



@app.route("/fr", methods = ["POST"])
def Fr_translator():
	return redirect(url_for("translator", output_language='French', text=request.form['text']), code=307)



if __name__ == "__main__":
	
	
	app.run(host = '0.0.0.0', port = 5000)

