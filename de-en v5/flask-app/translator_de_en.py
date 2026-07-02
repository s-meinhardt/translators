from flask import Flask, url_for, request, redirect, render_template, jsonify
from markupsafe import escape

import os
import json
import re
import requests
from math import log 

from nltk import sent_tokenize
from tokenizers import Tokenizer




# some parameters
INP_SEQ_LENGTH = 40
MODEL = {}
PORT = {}
HOST = {}

MODEL['de/en detect'] = 'detector'
HOST['de/en detect'] = 'http://localhost'   #  tensorflow serving requires this host
PORT['de/en detect'] = '8600'

MODEL['de-->en transl'] = 'translator'
HOST['de-->en transl'] = 'http://localhost'   #  tensorflow serving requires this host
PORT['de-->en transl'] = '8601'

MODEL['en-->de transl'] = 'translator'
HOST['en-->de transl'] = 'http://localhost'   #  tensorflow serving requires this host
PORT['en-->de transl'] = '8602'




LANGUAGES = ['German', 'English']




# loading the tokenizers
vocab_path = 'vocabularies'
de_tokenizer = Tokenizer.from_file(os.path.join(vocab_path, 'wmt19_de_32.json'))
en_tokenizer = Tokenizer.from_file(os.path.join(vocab_path, 'wmt19_en_32.json'))
de_en_tokenizer = Tokenizer.from_file(os.path.join(vocab_path, 'wmt19_de_en_32.json'))




def detect_language(text):
	
	# tokenizing the input sentences
	sequences = tokenize(text, de_en_tokenizer)


	# correcting the length as the detector requires INP_SEQ_LENGTH
	delta = INP_SEQ_LENGTH - len(sequences[0])
	sequences = [sequence+delta*[0] for sequence in sequences]


	# sending a detection request to a TensorFlow Server instance
	model = MODEL['de/en detect']
	host = HOST['de/en detect']
	port = PORT['de/en detect']
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






def tokenize(text, inp_tokenizer):
	
	# spliting the text into sentences
	sentences = sent_tokenize(text, language = 'german')
	

	# tokenizing the sentences and computing the max sequence length
	tmp_sequences = [inp_tokenizer.encode(sentence).ids for sentence in sentences]    
	max_length = min(max([len(sequence) for sequence in tmp_sequences]), INP_SEQ_LENGTH)  
  

	# breaking too long sequences and padding with zeros
	sequences = []
	for sequence in tmp_sequences:
		while len(sequence) > 0:
			start_sequence = sequence[: INP_SEQ_LENGTH]
			sequences.append(start_sequence + (max_length-len(start_sequence))*[0])
			sequence = sequence[INP_SEQ_LENGTH : ]


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






def translate(text, input_language):

	# choosing tokenizers and server port according to the input language
	if input_language == 'German':
		output_language = 'English'
		model = MODEL['de-->en transl']
		host = HOST['de-->en transl']
		port = PORT['de-->en transl']
		inp_tokenizer = de_tokenizer 
		outp_tokenizer = en_tokenizer
	elif input_language == 'English':
		output_language = 'German'
		model = MODEL['en-->de transl']
		host = HOST['en-->de transl']
		port = PORT['en-->de transl']
		inp_tokenizer = en_tokenizer
		outp_tokenizer = de_tokenizer   


	# sending a translation request to a TensorFlow Server instance
	server_url = f'{host}:{port}/v1/models/{model}:predict'
	headers = {"content-type": "application/json"}
	data = {"signature_name": "serving_default",
			"instances": tokenize(text, inp_tokenizer)}
	response = requests.post(server_url, json = data, headers = headers)
	response.raise_for_status()   
	
 
	# detokenizing the sequences and computing the probability and the number of all tokens
	# Note: TensorFlow Server returns batches of dictionaries and not dictionary of batches
	translation = ''
	probability = 1   
	n_tokens = 0
	for output in response.json()['predictions']:   	
		translation += outp_tokenizer.decode(output['sequences']) + ' '
		probability *= output['probabilities']
		n_tokens += len(output['sequences'])


	# cleaning the translation
	if output_language == 'German':
		translation = clean_translation_en(text, translation)
	elif output_language == 'English':
		translation = clean_translation_de(text, translation)


	# computing the confidence as the (geometric) average of the token probabilities
	confidence = probability**(1. / n_tokens)    
	confidence = int(confidence*100)	# normalizing the confidence


	return translation, confidence, output_language





app = Flask(__name__)



@app.route("/")
def home():
	return redirect(url_for("translator"))





@app.route("/translator", methods = ["POST", "GET"])
def translator():
	text = ''
	translation = ''
	confidence = 0
	input_language = ''
	output_language = ''
	if request.method == "GET":
		return render_template("translator.html", translation = translation, confidence = confidence, text = text, input_language = input_language, output_language = output_language)

	elif request.method == "POST":
		text = request.form["text"]
		input_language = detect_language(text)
		translation, confidence, output_language = translate(text, input_language)		

	return render_template("translator.html", translation = translation, confidence = confidence, text = text, input_language = input_language, output_language = output_language)






@app.route("/prediction", methods=["POST"])
def predict():
	if request.method == "POST":		
		text = request.get_json()
		if not isinstance(text, str):
			return f'Input is of type {type(text)} which is not supported!'
		
		input_language = detect_language(text)
		translation, confidence, output_language = translate(text)
				
	return json.dumps({"translation" : translation, "confidence" : confidence, "input language": input_language, "output language": output_language})

 




if __name__ == "__main__":
	
	
	app.run(host = '0.0.0.0', port = 5000)

