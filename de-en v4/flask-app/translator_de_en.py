from flask import Flask, url_for, request, redirect, render_template, jsonify
from markupsafe import escape

import os
import json
import re

import tensorflow as tf

from nltk import sent_tokenize
from utils.text.tokenizer import BertWordPieceTokenizer as Tokenizer


tf.config.set_visible_devices([], 'GPU')


# some parameters
INP_SEQ_LENGTH = 40
HISTORY_LENGTH = 40
VERSION = 8


#loading the beam search model
model_path = os.path.join('models', f'{VERSION:04d}')
beam_search_loaded = tf.saved_model.load(model_path)
beam_search = beam_search_loaded.signatures['serving_default']  # to avoid errors, we need to specify the concrete serving function in a second step
																	# we also need a different name to avoid confusion with the function defined below


# loading the tokenizers
vocab_path = os.path.join('models', f'{VERSION:04d}', 'vocabularies')
inp_tokenizer = Tokenizer(directory = vocab_path, model_prefix = 'wmt19_de_32')
outp_tokenizer = Tokenizer(directory = vocab_path, model_prefix = 'wmt19_en_32')



def tokenize(text):
    
    sentences = sent_tokenize(text, language = 'german')
    
    print('\nThe original sentences:')
    print(sentences)

    tmp_sequences = [inp_tokenizer.tokenize(sentence) for sentence in sentences]    
    max_length = min(max([len(sequence) for sequence in tmp_sequences]), INP_SEQ_LENGTH)  
    print(f'\nmax_length = {max_length}')

    print('\nThe original tokenized sentences:')
    print(tmp_sequences)
    print('\nThe original detokenized tokens:')
    for sequence in tmp_sequences:
    	print(inp_tokenizer.detokenize(sequence)+'\n\n')


    sequences = []
    for sequence in tmp_sequences:
        while len(sequence) > 0:
            start_sequence = sequence[: INP_SEQ_LENGTH]
            sequences.append(start_sequence + (max_length-len(start_sequence))*[0])
            sequence = sequence[INP_SEQ_LENGTH : ]

    print(f'\nThe tokens after being split up to match INP_SEQ_LENGTH={INP_SEQ_LENGTH}')
    print(sequences)
    print('\nThe detokenize splitted up tokens')
    for sequence in sequences:
    	print(inp_tokenizer.detokenize(sequence)+'\n\n')

    return tf.constant(sequences)




def clean_translation(text, translation):
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




def translate(text):    
    
    outputs = beam_search(tokenize(text))
    sequences, probabilities = outputs['sequences'], outputs['probabilities']

    print('\nThe tokens of the translation:')
    print(sequences)
	#for sequence in sequences.numpy().tolist():
	#	print(len(sequence))

    translations = [outp_tokenizer.detokenize(sequence) for sequence in sequences.numpy()]
    
    translation = ''
    for i, transl in enumerate(translations):
    	#transl = transl[0].upper() + transl[1:]
    	translation += transl + ' '

    translation = clean_translation(text, translation)
        
    n_tokens = 0
    cum_probability = 1   # cumulated probability
    for sequence, probability in zip(sequences.numpy(), probabilities.numpy()):
    	n_tokens += len(sequence)
    	cum_probability *= probability

    confidence = cum_probability**(1. / n_tokens)
    # normalizing the confidence
    confidence = int(confidence*100)

    return translation, confidence





app = Flask(__name__)



@app.route("/")
def home():
	return redirect(url_for("translator"))



@app.route("/translator", methods = ["POST", "GET"])
def translator():
	text = ''
	translation = ''
	confidence = 0
	if request.method == "GET":
		return render_template("translator.html", translation = translation, confidence = confidence, text = text)

	elif request.method == "POST":
		text = request.form["text"]
		translation, confidence = translate(text)		

	return render_template("translator.html", translation = translation, confidence = confidence, text = text)



@app.route("/prediction", methods=["POST"])
def predict():
	if request.method == "POST":		
		text = request.get_json()
		if not isinstance(text, str):
			return f'Input is of type {type(text)} which is not supported!'
		
		translation, confidence = translate(text)
				
	return json.dumps({"translation" : translation, "confidence" : confidence})

 


if __name__ == "__main__":
	
	
	app.run(host = '0.0.0.0', port = 5000)

