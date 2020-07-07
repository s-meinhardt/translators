import tensorflow as tf
import tensorflow_datasets as tfds
from flask import Flask, url_for, request, redirect, render_template, jsonify
from markupsafe import escape
from utils import BeamSearch
import os
import json


tf.config.set_visible_devices([], 'GPU')


def load_beam_search():

	loaded_model = tf.saved_model.load('model')

	# We need to wrap the loaded model in a tf.function to avoid unnecessary tracing
	input_signature = [{'encoder_input': tf.TensorSpec(shape = [None, None], dtype = tf.int32, name = 'encoder_input'),
						'decoder_input': tf.TensorSpec(shape = [None, None], dtype = tf.int32, name = 'decoder_input')}]
		
	@tf.function(experimental_relax_shapes = True, input_signature = input_signature)
	def wrapped_model(inputs):
		print(f"Tracing with encoder input = {inputs['encoder_input']} and decoder input = {inputs['decoder_input']}")
		return loaded_model(inputs, training = False)

	# We wrap again to allow a 'training = False' argument in BeamSearch.__call__() algorithm 
	def transformer(inputs, **kwargs):
		return wrapped_model(inputs)

	inp_tokenizer_file = os.path.join('tokenizer', 'input_tokenizer')
	outp_tokenizer_file = os.path.join('tokenizer', 'output_tokenizer')

	# creating a dictionary of tokenizers
	tokenizer = {}
	tokenizer['input'] = tfds.features.text.SubwordTextEncoder.load_from_file(inp_tokenizer_file)
	tokenizer['output'] = tfds.features.text.SubwordTextEncoder.load_from_file(outp_tokenizer_file)

	#global beam_search
	beam_search = BeamSearch(transformer, tokenizer, beam_width = 3, max_seq_length = 40)

	return beam_search


beam_search = load_beam_search()

app = Flask(__name__)



@app.route("/")
def home():
	return redirect(url_for("translator"))



@app.route("/translator", methods = ["POST", "GET"])
def translator():
	sentence = ''
	translation = ''
	probability = 0.0
	if request.method == "GET":
		return render_template("translator.html", translation = translation, probability = probability, sentence = sentence)

	elif request.method == "POST":
		sentence = request.form["sentence"]
		inp_sentence = sentence.lower().strip()

		translation, probability = beam_search.translate(inp_sentence)
		
		# formating the results
		translation = translation.numpy().decode()
		translation = translation[0].upper() + translation[1:]
		probability = round(float(100*probability),3)

	return render_template("translator.html", translation = translation, probability = probability, sentence = sentence)



@app.route("/prediction", methods=["POST"])
def predict():
	if request.method == "POST":
		
		sentence = request.get_json()

		if not isinstance(sentence, str):
			return f'Input is of type {type(sentence)} which is not supported!'

		inp_sentence = sentence.lower().strip()

		translation, probability = beam_search.translate(inp_sentence)
		
		# formating the results to be jsonizable
		translation = translation.numpy().decode()
		translation = translation[0].upper() + translation[1:]
		probability = round(float(100*probability),3)
		
	return json.dumps({"translation" : translation, "probability" : probability})

 


if __name__ == "__main__":
	
	
	app.run(host = '0.0.0.0', port = 5000)

