import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds
#import tensorflow_text as text
import tokenizers
from sentencepiece import SentencePieceTrainer

class TFSentencePieceTokenizer(object):
    


    
    def __init__(self, directory, model_prefix, out_type = tf.int32, **kwargs):
        super(TFSentencePieceTokenizer, self).__init__(**kwargs)  
        self.model_saved = False
        self.name = 'TFSentencePieceTokenizer'
        self.model_prefix = model_prefix
        self.out_type = out_type
        try:
#            self.tokenizer = text.SentencepieceTokenizer()
            pass
        except:
            print("Cannot build tokenizer. Please run: pip install tensorflow-text \n")
        try:
            self.model_path = os.path.join(directory, model_prefix + '.json')
        except:
            print("Error: Make sure that 'directory' and 'model_prefix are strings!\n")
        try:
            self._build(self.model_path, self.out_type)
        except:
            print(f"Cannot find a model file at '{self.model_path}'.") 
            print("Check 'directory' and 'model_prefix' arguments or call the 'train' method to build a tokenizer!\n")
        
        
        
    def train(self, text_file, vocab_size = 32000, save_tokenizer = True):
        if not os.path.exists(text_file):
            print("Error: '" + text_file + "' does not exist.")
            pass
        try:            
            SentencePieceTrainer.train(model_prefix=self.model_prefix, 
                                       input=text_file, 
                                       vocab_size=vocab_size)
        except:
            print("Error: Training failed! Make sure that each line in the text file is a single sentence.")
            print("Also make sure that 'sentencepiece' is installed. If not, run: pip install sentencepiece")
        try:
            tmp_model_path = self.model_prefix + '.model'
            self._build(tmp_model_path, self.out_type)
        except:
            print('Error: Cannot get vocab size.')
        if save_tokenizer:
            try:
                shutil.copy2(tmp_model_path, self.model_path)
                self.tokenizer_saved = True
                print('Saved tokenizer at' + self.model_path)
            except:
                print('Error: Cannot save the model!')
        os.remove(tmp_model_path)
        
        
        
    def _build(self, model_path, out_type = tf.int32):        
#        self.tokenizer = text.SentencepieceTokenizer(model = tf.io.gfile.GFile(model_path, 'rb').read(), 
#                                                     out_type = out_type)
        self.vocab_size = self.tokenizer.vocab_size()
        
           
            
    def tokenize(self, sentence):
        return self.tokenizer.tokenize(tf.constant(sentence)).numpy()
    
    
    
    def tf_tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)
    
    
    
    def detokenize(self, tokens):
        return self.tokenizer.detokenize(tokens).numpy().decode()
    
    
    
    
        

    


class SentencePieceBPETokenizer(object):
    
    def __init__(self, directory, model_prefix = None, **kwargs):
        super(SentencePieceBPETokenizer, self).__init__(**kwargs)
        self.model_saved = False
        self.name = 'SentencePieceBPETokenizer'
        try:
            self.tokenizer = tokenizers.SentencePieceBPETokenizer()
        except:
            print("Cannot build tokenizer. Please run: pip install tokenizers \n")
        try:
            self.model_path = os.path.join(directory, model_prefix + '.json')
        except:
            print("Error: Make sure that 'directory' and 'model_prefix are strings!\n")
        try:
            self.tokenizer = tokenizers.Tokenizer.from_file(self.model_path)
            self.vocab_size = self.tokenizer.get_vocab_size()
        except:
            print(f"Cannot find a model file at '{self.model_path}'.") 
            print("Check 'directory' and 'model_prefix' arguments or call the 'train' method to build a tokenizer!\n")
        
        
                        
    def train(self, text_file, vocab_size = 32000, save_tokenizer = True):
        if not os.path.exists(text_file):
            print("Error: '" + text_file + "' does not exist.")
            pass
        try:
            self.tokenizer = tokenizers.SentencePieceBPETokenizer()
            self.tokenizer.train(files = text_file, vocab_size = vocab_size)
            print('trained')
        except:
            print('Error: Training failed! Make sure that each line in the text file is a single sentence.')
        try:
            self.vocab_size = self.tokenizer.get_vocab_size()
        except: 
            print('Error: Cannot get vocab size.')
        if save_tokenizer:
            try:
                self.tokenizer.save(self.model_path, pretty=True)
                self.tokenizer_saved = True
                print('Saved tokenizer at' + self.model_path)
            except:
                print('Error: Cannot save the model!')
        
        
        
    def tokenize(self, sentence):
        return self.tokenizer.encode(sentence).ids
    
    
    
    def tf_tokenize(self, sentence):
        """Wrapping the non-Tensorflow 'tokenize' function into a 'tf.py_function' 
           turning it into a Tensorflow function which can be used in graph mode."""
        
        # don't forget 'dtype=tf.int32' otherwise get an error for sequences of length 0 
        tokens =  tf.py_function(func = lambda sentence: tf.constant(
                                 self.tokenize(sentence.numpy().decode()), dtype=tf.int32), 
                                 inp = [sentence], Tout = tf.int32)        
        tokens.set_shape([None])   # The length depends on the sentence
                                   # We provide a 'shape' argument to avoid errors      
        return tokens
        
        
        
    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens)

        

        
        
        
        
        
        
class BertWordPieceTokenizer(object):
    
    def __init__(self, directory, model_prefix = None, **kwargs):
        super(BertWordPieceTokenizer, self).__init__(**kwargs)
        self.model_saved = False
        self.name = 'BertWordPieceTokenizer'
        try:
            self.tokenizer = tokenizers.BertWordPieceTokenizer()
        except:
            print("Cannot build tokenizer. Please run: pip install tokenizers \n")
        try:
            self.model_path = os.path.join(directory, model_prefix + '.json')
        except:
            print("Error: Make sure that 'directory' and 'model_prefix are strings!\n")
        try:
            self.tokenizer = tokenizers.Tokenizer.from_file(self.model_path)
            self.vocab_size = self.tokenizer.get_vocab_size()
        except:
            print(f"Cannot find a model file at '{self.model_path}'.") 
            print("Check 'directory' and 'model_prefix' arguments or call the 'train' method to build a tokenizer!\n")
        
              
            
    def train(self, text_file, vocab_size = 32000, save_tokenizer = True, lowercase = True):
        if not os.path.exists(text_file):
            print("Error: '" + text_file + "' does not exist.")
            pass
        try:
            self.tokenizer = tokenizers.BertWordPieceTokenizer(lowercase = lowercase)
            self.tokenizer.train(files = text_file, vocab_size = vocab_size)
            print('trained')
        except:
            print('Error: Training failed! Make sure that each line in the text file is a single sentence.')
        try:
            self.vocab_size = self.tokenizer.get_vocab_size()
        except: 
            print('Error: Cannot get vocab size.')
        if save_tokenizer:
            try:
                self.tokenizer.save(self.model_path, pretty=True)
                self.tokenizer_saved = True
                print('Saved tokenizer at' + self.model_path)
            except:
                print('Error: Cannot save the model!')
        
        
        
    def tokenize(self, sentence):
        return self.tokenizer.encode(sentence).ids
    
    
    
    
    def tokenize_batch(self, batch):
        return self.tokenizer.encode_batch(batch)
    
#    def tokenize_batch(self, batch):
#        if isinstance(batch, type(tf.constant([b'x', b'y'], dtype=tf.string))):
#            batch = batch.numpy()
#            batch = [line.decode() for line in batch]
#            
#        encoded_batch = self.tokenizer.encode_batch(batch)
#        sequences = [encoded_line.ids for encoded_line in encoded_batch]
        
#        return tf.data.Dataset.from_tensor_slices(sequences)
    
    
    
    def tf_tokenize(self, sentence):
        """Wrapping the non-Tensorflow 'tokenize' function into a 'tf.py_function' 
           turning it into a Tensorflow function which can be used in graph mode."""
        
        # don't forget 'dtype=tf.int32' otherwise get an error for sequences of length 0 
        tokens =  tf.py_function(func = lambda sentence: tf.constant(
                                 self.tokenize(sentence.numpy().decode()), dtype=tf.int32), 
                                 inp = [sentence], Tout = tf.int32)        
        tokens.set_shape([None])   # The length depends on the sentence
                                   # We provide a 'shape' argument to avoid errors      
        return tokens
    

    
    def tf_tokenize_batch(self, batch):
        """Wrapping the non-Tensorflow 'tokenize' function into a 'tf.py_function' 
           turning it into a Tensorflow function which can be used in graph mode."""
        
        # don't forget 'dtype=tf.int32' otherwise get an error for sequences of length 0 
        tokens =  tf.py_function(func = lambda batch: self.tokenize_batch(batch), 
                                 inp = [batch], Tout = tf.int32)        
        #tokens.set_shape([None])   # The length depends on the sentence
                                   # We provide a 'shape' argument to avoid errors      
        return tokens
        
        
        
    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens)
    
    
    
    def tf_detokenize(self, sequence):
        sentences = tf.py_function(func = lambda sequence: tf.constant(
                                   self.detokenize(sequence.numpy()), tf.string),
                                   inp = [sequence], Tout=tf.string)
        #sentences.set_shape([None])
        
        return sentences
    
    
    

    
    
    
class SubwordTextEncoder(object):
  
    def __init__(self, directory, model_prefix, **kwargs):
        super(SubwordTextEncoder, self).__init__(**kwargs)
        self.model_saved = False
        self.name = 'SubwordTextEncoder'
        try:
            self.tokenizer = tfds.features.text.SubwordTextEncoder()
        except:
            print("Cannot build tokenizer. Please run: pip install tensorflow-datasets \n")
        try:
            self.model_path = os.path.join(directory, model_prefix + '.subwords')
        except:
            print("Error: Make sure that 'directory' and 'model_prefix are strings!\n")
        try:                
            self.tokenizer.load_from_file(self.model_path)
            self.vocab_size = self.tokenizer.vocab_size
        except:
            print(f"Cannot find a model file at '{self.model_path}'.") 
            print("Check 'directory' and 'model_prefix' arguments or call the 'train' method to build a tokenizer!\n")
        
            
            
    def train(self, text_file, vocab_size = 32000, save_tokenizer = True):
        if not os.path.exists(text_file):
            print("Error: '" + text_file + "' does not exist.")
            pass
        try:
            self.tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(text_file)
            self.vocab_size = self.tokenizer.vocab_size
        except:
            print("Error: Training failed! Make sure that 'text_file' is a string generator with each line being a single sentence.")
        if save_tokenizer:
            try:
                self.tokenizer.save_to_file(self.model_path)
                self.tokenizer_saved = True
                print('Saved tokenizer at' + self.model_path)
            except:
                print('Error: Cannot save the model!')
      
    
        
    def tokenize(self, sentence):
        return self.tokenizer.encode(sentence)
    
    
       
    def tf_tokenize(self, sentence):
        """Wrapping the non-Tensorflow 'tokenize' function into a 'tf.py_function' 
           turning it into a Tensorflow function which can be used in graph mode."""
        
        # don't forget 'dtype=tf.int32' otherwise get an error for sequences of length 0 
        tokens =  tf.py_function(func = lambda sentence: tf.constant(
                                 self.tokenize(sentence.numpy().decode()), dtype=tf.int32), 
                                 inp = [sentence], Tout = tf.int32)        
        tokens.set_shape([None])   # The length depends on the sentence
                                   # We provide a 'shape' argument to avoid errors      
        return tokens
    

    
    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens).numpy().decode()
    
    
    
    
    
    
class BertWordPieceTokenizer2(object):
    
    def __init__(self, directory, model_prefix = None, **kwargs):
        super(BertWordPieceTokenizer2, self).__init__(**kwargs)
        self.model_saved = False
        self.directory = directory
        self.name = 'BertWordPieceTokenizer2'
        try:
            self.model_path = os.path.join(directory, model_prefix + '.txt')
        except:
            print("Error: Make sure that 'directory' and 'model_prefix are strings!\n")
            return
        try:
            self.tokenizer = tokenizers.BertWordPieceTokenizer(vocab_file = self.model_path)
            self.vocab_size = self.tokenizer.get_vocab_size()
        except:
            print(f"Cannot build tokenizer. If there is no vocabulary file at '{self.model_path},")
            print("run the train method. Moreover, make sure that 'tokenizers' is installed.") 
            print("If not, run: pip install tokenizers \n")
        
              
            
    def train(self, text_file, vocab_size = 32000, save_tokenizer = True):
        if not os.path.exists(text_file):
            print("Error: '" + text_file + "' does not exist.")
            pass
        try:
            self.tokenizer = tokenizers.BertWordPieceTokenizer()
            self.tokenizer.train(files = text_file, vocab_size = vocab_size)
            print('trained')
        except:
            print('Error: Training failed! Make sure that each line in the text file is a single sentence.')
        try:
            self.vocab_size = self.tokenizer.get_vocab_size()
        except: 
            print('Error: Cannot get vocab size.')
        if save_tokenizer:
            try:
                self.tokenizer.save_model(self.directory)
                os.rename(os.path.join(self.directory, 'vocab.txt'), self.model_path)
                self.tokenizer_saved = True
                print('Saved tokenizer at' + self.model_path)
            except:
                print('Error: Cannot save the model!')
        
        
        
    def tokenize(self, sentence):
        return self.tokenizer.encode(sentence).ids[1:-1]
#        if len(tokens) == 0:
#            tokens = [-1]
#        return tokens
    
    
    
    def tf_tokenize(self, sentence):
        """Wrapping the non-Tensorflow 'tokenize' function into a 'tf.py_function' 
           turning it into a Tensorflow function which can be used in graph mode."""
        
        # don't forget 'dtype=tf.int32' otherwise get an error for sequences of length 0 
        tokens =  tf.py_function(func = lambda sentence: tf.constant(
                                 self.tokenize(sentence.numpy().decode()), dtype=tf.int32), 
                                 inp = [sentence], Tout = tf.int32)        
        tokens.set_shape([None])   # The length depends on the sentence
                                   # We provide a 'shape' argument to avoid errors      
        return tokens
        
        
        
    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens)
    