import os
import shutil



class TFSentencePieceTokenizer(object):
    import tensorflow as tf
    
    def __init__(self, directory, model_prefix, out_type = tf.int32, **kwargs):
        super(TFSentencePieceTokenizer, self).__init__(**kwargs)        
        if isinstance(model_prefix, str) and os.path.exists(directory):
            self.model_path = os.path.join(directory, model_prefix + '.model' )
            self.model_prefix = model_prefix
            self.out_type = out_type
            try:
                self._build(self.model_path, self.out_type)
            except:
                print(f"Cannot find a model file at '{self.model_path}'.") 
                print("Check 'directory' and 'model_prefix' arguments or call the 'train' method to build a tokenizer!\n")
        else:
            print("Error: 'directory' does not exist or 'model_prefix' is not a string!")
        
        
    def train(self, text_file, vocab_size = 32000, save_tokenizer = True):
        if not os.path.exists(text_file):
            print("Error: '" + text_file + "' does not exist.")
            pass
        try:
            import sentencepiece as spm
            spm.SentencePieceTrainer.train(model_prefix=self.model_prefix, 
                                           input=text_file, 
                                           vocab_size=vocab_size)
        except:
            print('Error: Training failed! Make sure that each line in the text file is a single sentence.')
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
        import tensorflow_text as text
        self.tokenizer = text.SentencepieceTokenizer(model = tf.io.gfile.GFile(model_path, 'rb').read(), 
                                                     out_type = out_type)
        self.vocab_size = self.tokenizer.vocab_size()
        self.name = 'TFSentencePieceTokenizer'
        
        
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
        self.model_path = os.path.join(directory, model_prefix + '.json' )
        if isinstance(model_prefix, str) and os.path.exists(directory):
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(self.model_path)
                self.vocab_size = self.tokenizer.get_vocab_size()
                self.name = 'SentencePieceBPETokenizer'
            except:
                print(f"Cannot find a model file at '{self.model_path}'.") 
                print("Check 'directory' and 'model_prefix' arguments or call the 'train' method to build a tokenizer!\n")
        else:
            print("Error: 'directory' does not exist or 'model_prefix' is not a string!")
            
            
    def train(self, text_file, vocab_size = 32000, save_tokenizer = True):
        if not os.path.exists(text_file):
            print("Error: '" + text_file + "' does not exist.")
            pass
        try:
            from tokenizers import SentencePieceBPETokenizer 
            self.tokenizer = SentencePieceBPETokenizer()
            self.name = 'SentencePieceBPETokenizer'
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
        import tensorflow as tf
        tokens =  tf.py_function(func = lambda sentence: tf.constant(self.tokenize(sentence.numpy().decode())), 
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
        self.model_path = os.path.join(directory, model_prefix + '.json' )
        if isinstance(model_prefix, str) and os.path.exists(directory):
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(self.model_path)
                self.vocab_size = self.tokenizer.get_vocab_size()
                self.name = 'BertWordPieceTokenizer'
            except:
                print(f"Cannot find a model file at '{self.model_path}'.") 
                print("Check 'directory' and 'model_prefix' arguments or call the 'train' method to build a tokenizer!\n")
        else:
            print("Error: 'directory' does not exist or 'model_prefix' is not a string!")
            
            
    def train(self, text_file, vocab_size = 32000, save_tokenizer = True):
        if not os.path.exists(text_file):
            print("Error: '" + text_file + "' does not exist.")
            pass
        try:
            from tokenizers import BertWordPieceTokenizer 
            self.tokenizer = BertWordPieceTokenizer()
            self.name = 'BertWordPieceTokenizer'
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
        import tensorflow as tf
        tokens =  tf.py_function(func = lambda sentence: tf.constant(self.tokenize(sentence.numpy().decode())), 
                                 inp = [sentence], Tout = tf.int32)        
        tokens.set_shape([None])   # The length depends on the sentence
                                   # We provide a 'shape' argument to avoid errors      
        return tokens
        
        
    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens)