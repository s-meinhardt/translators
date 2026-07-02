import tensorflow as tf

class BeamSearch(tf.Module):
    
    def __init__(self, model, tokenizer, beam_width, max_seq_length, **kwargs):
        super(BeamSearch, self).__init__(**kwargs)
        self.model = model
        self.inp_tokenizer = tokenizer['input']
        self.outp_tokenizer = tokenizer['output']
        self.inp_vocab_size = self.inp_tokenizer.vocab_size
        self.outp_vocab_size = self.outp_tokenizer.vocab_size
        self.beam_width = beam_width
        self.max_seq_length = max_seq_length
        self.stop_words = ['.', ' .', ' . ', '!', ' !', ' ! ', '?', ' ?', ' ? ']
        self.stop_codes = [0] + list(map(lambda x: self.outp_tokenizer.encode(x)[0], self.stop_words))
    
    
        
    def __call__(self, sentence):
    
        # the probability distribution choosing <end> with certainty
        end_distribution = tf.constant([1.] + self.outp_vocab_size*[0.], shape=[1, self.outp_vocab_size + 1])
    
        # add a <start> and <end> token to the input sentence and encode the result
        enc_inp_sentence = tf.constant([self.inp_vocab_size] + self.inp_tokenizer.encode(sentence) + [0])    
        enc_inp_sentence = enc_inp_sentence[tf.newaxis, : ]  # add a batch dimension
    
        # the encoded sentences
        # the start sentence is just <start>
        decoder_input = tf.constant(self.outp_vocab_size, shape=[1,1], dtype=tf.int32) 
    
        # the probability of our sentences
        # the start sentence has probability one
        proba = tf.ones(shape=[1, 1], dtype=tf.float32)  
    
        for i in range(1, self.max_seq_length):
        
            # is the previous sentence ending with <end>
            condition = tf.map_fn(lambda x: x in self.stop_codes, decoder_input[:, -1], dtype=tf.bool)[ : , tf.newaxis] 
        
            # if all sentences end with <end>, stop the iteration
            if tf.reduce_all(condition == True):
                break
        
            # repeat the input sentence as many times as given by the batch size of decoder_input
            encoder_input = tf.repeat(enc_inp_sentence, repeats = decoder_input.shape[0], axis=0)
    
            predictions = self.model({'encoder_input': encoder_input, 'decoder_input': decoder_input}, training = False)

            # compute the conditional probabilities for the next word 
            # do this for each previous sentence 
            probabilities = tf.nn.softmax(predictions[:, -1, :], axis = -1)  # (beam_width, output_vocab_size)
                
            # if previous sentence ends with <end>, pic end_distribution to choose <end> again
            probabilities = tf.where(condition, end_distribution, probabilities)
        
            # choose the highes conditional probabilities for each previous sentence
            top_cond_proba, top_words = tf.math.top_k(probabilities, k = self.beam_width)  # (beam_width, beam_width)
        
            # compute the probabilities for the new sentences and flatten the results
            top_proba = tf.reshape(proba[:,tf.newaxis] * top_cond_proba, shape = [-1]) 
        
            # flatten the lists of best next words
            top_words = tf.reshape(top_words, shape = [-1])
            
            # choose among all sentence probabilities the best beam_width
            proba, top_idx = tf.math.top_k(top_proba, k = self.beam_width)
    
            # pic the associated best next words
            top_words = tf.gather(params = top_words, indices = top_idx) 

            # pic the previous sentences giving rise to the best next words
            top_dec_inputs = tf.gather(params = decoder_input, indices = top_idx // self.beam_width)
        
            # extend the previous sentences with their best next words
            decoder_input = tf.concat([ top_dec_inputs, top_words[:, tf.newaxis] ], axis = -1)
            
        translations= tf.map_fn(lambda x: self.outp_tokenizer.decode(x), decoder_input[:,1:], dtype=tf.string)
        
        return [translations, proba] 
        #return decoder_input
    
    
    
    def translations(self, sentence):
        # compute self.beam_width many translations
        translations, probabilities = self.__call__(sentence)
        
        print(f'Input: {sentence}\n')
        for i, translation in enumerate(translations):
            print(f'Translation {i+1} with probability {100*probabilities[i]:.3f}%')
            print(translation.numpy().decode() + '\n')
        
    
    
    def translate(self, sentence):
        # compute self.beam_width many translations
        translations, probabilities = self.__call__(sentence)
        
        # pick the best and return it
        best_idx = tf.argmax(probabilities).numpy()
        return [translations[best_idx], probabilities[best_idx]]
    
