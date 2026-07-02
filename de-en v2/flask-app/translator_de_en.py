import tensorflow as tf
import tensorflow_datasets as tfds
from flask import Flask, url_for, request, redirect, render_template, jsonify
from markupsafe import escape
#from sequences import BeamSearch
import os
import json

from nltk import sent_tokenize


from utils.layers import MHA, MHAU, PositionalEmbedding
from utils.metrics import Weighted_Loss, Factor
from utils.callbacks import My_Callback
from utils.sequences import BeamSearch
from utils.models.transformer import CustomSchedule
from utils.text.tokenizer import SentencePieceBPETokenizer as Tokenizer


tf.config.set_visible_devices([], 'GPU')

# some parameters
INP_SEQ_LENGTH = 40
HISTORY_LENGTH = 40
VERSION = 1



# loading the Markov kerneln
transformer = tf.keras.models.load_model(os.path.join('models', f'{VERSION:04d}'), 
					custom_objects = {'MHA': MHA, 
				                      'MHAU': MHAU, 
				                      'PositionalEmbedding': PositionalEmbedding, 
				                      'Weighted_Loss': Weighted_Loss,
				                      'Factor': Factor,
				                      'CustomSchedule': CustomSchedule,
				                      'My_Callback': My_Callback})



# loading the tokenizers
vocab_path = os.path.join('models', f'{VERSION:04d}', 'vocabularies')
inp_tokenizer = Tokenizer(directory = vocab_path, model_prefix = 'wmt19_inp_tokenizer')
outp_tokenizer = Tokenizer(directory = vocab_path, model_prefix = 'wmt19_outp_tokenizer')



#loading beam search
BEAM_WIDTH = 5
MAX_SEQ_LENGTH = 100     # the maximal length of translations, should be greater than INP_SEQ_LENGTH  
stop_words = ['.', ' .', ' . ', '!', ' !', ' ! ', '?', ' ?', ' ? ']
stop_codes = list(map(lambda x: outp_tokenizer.tokenize(x)[0], stop_words))

beam_search = BeamSearch(model = transformer, 
	                beam_width = BEAM_WIDTH, 
	                history_length = HISTORY_LENGTH, 
	                max_seq_length = MAX_SEQ_LENGTH, 
	                start_code = outp_tokenizer.vocab_size, 
	                end_code = 0,   # must be zero
	                stop_codes = stop_codes, 
	                num_codes = outp_tokenizer.vocab_size+2)   # two extra codes for <start> and <end>




def tokenize(text):
    
    sentences = sent_tokenize(text, language = 'german')
    
    tmp_sequences = [inp_tokenizer.tokenize(sentence) for sentence in sentences]    
    max_length = min(max([len(sequence) for sequence in tmp_sequences]), INP_SEQ_LENGTH)
           
    sequences = []
    for sequence in tmp_sequences:
        while len(sequence) > 0:
            start_sequence = sequence[: INP_SEQ_LENGTH]
            sequences.append(start_sequence + (max_length-len(start_sequence))*[0])
            sequence = sequence[INP_SEQ_LENGTH : ]

    return tf.constant(sequences)




def translate(text):    
    
    outputs = beam_search(tokenize(text))
    sequences, probabilities = outputs['sequences'], outputs['probabilities']
    translations = [outp_tokenizer.detokenize(sequence) for sequence in sequences.numpy()]
    
    translation = ''
    for i, transl in enumerate(translations):
    	#transl = transl[0].upper() + transl[1:]
    	translation += transl + ' '
        
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

