from flask import Flask, url_for, request, redirect, render_template, jsonify
from markupsafe import escape

import os
import json

import tensorflow as tf

from nltk import sent_tokenize



# model version
VERSION = 8


#loading the translator model
model_path = os.path.join('models', f'{VERSION:04d}')
translator_loaded = tf.saved_model.load(model_path)
de_en_translator = translator_loaded.signatures['serving_default']  # to avoid errors, we need to specify the concrete serving function in a second step
																	# we also need a different name to avoid confusion with the function defined below


def translate(text):    
    
    results = de_en_translator(tf.constant(sent_tokenize(text, language = 'german')))
    translations, probabilities = results['translations'], results['probabilities']
    
    translation = ''
    for i, transl in enumerate(translations.numpy()):
        translation += transl.decode() + ' ' #  f' ({probabilities.numpy()[i]*100:.2f}%) ' 
        
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

