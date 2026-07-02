import tensorflow as tf






class BeamSearch(tf.Module):
    
    def __init__(self, model, beam_width, history_length, max_seq_length, start_code, stop_codes, num_codes, end_code = 0, **kwargs):
        
        super(BeamSearch, self).__init__(**kwargs)
        self.model = model                           # the Markov kernel
        self.history_length = history_length         # the maximal number of codes/states used by the model to predict the next one
        self.beam_width = beam_width                 # the number of alternatives to consider when forming sequences of codes/states
        self.max_seq_length = max_seq_length         # the maximal length of the output sequences
        self.start_code = start_code                 # the code for the <start> state
        self.stop_codes = [end_code] + stop_codes    # codes for stop words and the <end> state
        self.num_codes = num_codes -1                # number of all codes/states except for the <start> code
        self.end_code = end_code                     # the code for the <end> state (states following the end of a sentence)
        
        
        # wrapping the self._call method into a tf.function for faster computation and serving
        @tf.function(input_signature = [tf.TensorSpec.from_tensor(model.input['encoder_input'])])
        def call(inputs):
            print('Tracing BeamSearch!\n')
            return self._call(inputs)
        
        
        self.call = call
        
    
        
        
    # making the class callable
    def __call__(self, inputs):
        return self.call(inputs)
        
       
    
    
    
    # this is the actual function performing beam search
    def _call(self, inputs):
                
        batch_size = tf.shape(inputs)[0]
        beam_size = 1  # will be changed later to beam_width
    
    
    
        # the probability distribution for the <end> code
        # extended by a batch and a beam dimension to be broadcastible         
        end_distribution = tf.sparse.to_dense(tf.SparseTensor(indices = [[0, 0, self.end_code]], 
                                                              values = [1.], 
                                                              dense_shape = [1, beam_size, self.num_codes]))
        
        # We change the probability '0' to some negative value to ensure that no code other than
        # the final code is chosen. This problem occurs if other very small probabilities 
        # are numerical identical to zero.
        end_distribution = tf.where(end_distribution == 0., -1.0e-15, end_distribution)
      
        
        
        # add another 'beam dimension' which will collecting the top beam_width extended sequences
        inputs = tf.expand_dims(inputs, axis = 1)
        
        
        
        # the output sequences, dtype = tf.int32 
        # shape = (batch_size, beam_size, current sequence length)
        # initialized with the output <start> code, repeated 'batch_size' times
        outp_sequences = tf.constant(self.start_code, shape = [1, beam_size, 1], dtype = tf.int32)
        outp_sequences = tf.repeat(outp_sequences, repeats = batch_size, axis = 0)


        
        # the probability of our output sequences, dtype = tf.float32
        # shape = (batch_size, beam_size)
        # initialized with probability one (per sequence)
        proba = tf.ones(shape=[batch_size, beam_size], dtype=tf.float32)

        
    
        for i in tf.range(1, self.max_seq_length):
            
            # as 'outp_sequences' and 'proba' change their first two dimensions 
            # in the loop, we need to include this command to avoid errors in graph mode
            tf.autograph.experimental.set_loop_options(
                    shape_invariants = [(outp_sequences, tf.TensorShape([None,None,None])),
                                        (proba, tf.TensorShape([None, None]))])
                    

                
            # the number of output sequences per input sequence 
            # should be beam_width after one iteration of the for loop
            beam_size = tf.shape(outp_sequences)[1]

            
            
            # check if we can stop the iteration
            # -------------------------------------------------------------------
            
            # repeated stop codes 
            # shape = (batch_size, beam_size, number of stop codes)
            stop_codes = tf.repeat([[self.stop_codes]], repeats = batch_size, axis = 0)
            stop_codes = tf.repeat([[self.stop_codes]], repeats = beam_size, axis = 1)
      
    
    
            # last codes 
            last_codes = outp_sequences[..., -1][...,tf.newaxis]    # (batch_size, beam_size, 1)
    
    
    
            # checking if the last code is in the list of stop codes, we do this by
            # comparing the set {last code} with the intersection of {last code} and {stop codes}
            # result is a boolean tensor of shape = (batch_size, beam_size, 1)
            condition = tf.reduce_any(last_codes == stop_codes, axis = -1)[..., tf.newaxis]
       
    
    
            # if all sequences end with a stop code, stop the 'for loop'
            if tf.reduce_all(condition):
                break
        
            
            
            
            # computing the next best codes and append them to their output sequences
            # -------------------------------------------------------------------------
            
            # repeat the inputs beam_size times along the beam dimension
            # and merge batch and beam dimension
            encoder_input = tf.repeat(inputs, repeats = beam_size, axis=1)
            encoder_input = tf.reshape(encoder_input, 
                                       shape = tf.concat([[-1], tf.shape(encoder_input)[2:]], axis=0))
            

            
            # the decoder input consists of the <start> codes and the last at most 
            # self.history_length - 1 many codes of the output sequences 
            if i < self.history_length:
                decoder_input = outp_sequences
            else:
                decoder_input = tf.concat([outp_sequences[:, :, :1], 
                                           outp_sequences[:, :, 1-self.history_length:]], 
                                           axis = -1)
            
            
            
            # merge the batch and the beam dimension
            decoder_input = tf.reshape(decoder_input, shape = [-1, tf.shape(decoder_input)[-1]])
           

            
            # computing the predictions using the model and reshaping the results to
            # (bach_size, beam_size, current sequence length, output_vocab_size)
            predictions = self.model({'encoder_input': encoder_input, 
                                      'decoder_input': decoder_input}, training=False)
            predictions = tf.reshape(predictions, shape = [batch_size, beam_size, -1, self.num_codes])
            
   

            # compute the conditional probabilities for the next code 
            # do this for each previous sentence 
            probabilities = tf.nn.softmax(predictions[:, :, -1, :], axis = -1)  # (bach_size, beam_size, output_vocab_size)
            
            
            
            # if previous sentence ends with <end>, pic end_distribution to choose <end> again
            probabilities = tf.where(condition, end_distribution, probabilities)
            

        
            # choose the highes conditional probabilities for each previous sequence
            top_cond_proba, top_codes = tf.math.top_k(probabilities, k = self.beam_width)  # (batch_size, beam_size, beam_width)
        

            
            # compute the probabilities for the new extended sequences and  
            # reshape the results by merging the last two dimensions
            top_proba = tf.reshape(proba[...,tf.newaxis] * top_cond_proba, shape = [batch_size, -1])   # (batch_size, beam_size*beam_width)
        
            
            
            # reshape the lists of best next codes by merging the last two dimensions
            top_codes = tf.reshape(top_codes, shape = [batch_size, -1])    # (batch_size, beam_size*beam_width)

           
        
            # choose among all sequence probabilities the best beam_width
            proba, top_idx = tf.math.top_k(top_proba, k = self.beam_width)    # (batch_size, beam_width)
            
    
    
            # pic the associated best next codes
            top_codes = tf.gather(params = top_codes, indices = top_idx, axis = 1, batch_dims = 1)     # (batch_size, beam_width)

            
            
            # pic the previous sequences giving rise to the best next codes
            # shape = (batch_size, beam_width, current sequence length)
            top_outp_sequences = tf.gather(params = outp_sequences, indices = top_idx // self.beam_width, axis= 1, batch_dims = 1)  
        
           
        
            # extend the previous sequences with their best next codes
            outp_sequences = tf.concat([ top_outp_sequences, top_codes[:, :, tf.newaxis] ], axis = -1)
            
            
            
        # returning only the best output sequences (without <start> code) 
        # note that proba is ordered descending in the beam dimension
        return {'sequences': outp_sequences[:, 0, 1:], 'probabilities': proba[:, 0]}   

    

    
    
    
    
    
    
    
    
class Translator(tf.Module):
    
    def __init__(self, 
                 tokenizer, 
                 kernel,                
                 beam_width, 
                 history_length,
                 inp_seq_length,
                 max_outp_seq_length, 
                 stop_words = ['.', ' .', ' . ', '!', ' !', ' ! ', '?', ' ?', ' ? '] 
                 ):
        
        self.inp_tokenizer = tokenizer['input']
        self.outp_tokenizer = tokenizer['output']
        self.kernel = kernel
        self.inp_seq_length = inp_seq_length
        
        stop_codes = list(map(lambda x: self.outp_tokenizer.tokenize(x)[0], stop_words))

        self.beam_search = BeamSearch(model = self.kernel, 
                                 beam_width = beam_width, 
                                 history_length = history_length, 
                                 max_seq_length = max_outp_seq_length, 
                                 start_code = self.outp_tokenizer.start_code, 
                                 end_code = self.outp_tokenizer.end_code,   # must be zero
                                 stop_codes = stop_codes, 
                                 num_codes = self.outp_tokenizer.vocab_size + 2)
    
    
    
    
    def _tokenize(self, sentence):
        sequence = self.inp_tokenizer.tf_tokenize(sentence)
        return tf.data.Dataset.from_tensor_slices(sequence).batch(self.inp_seq_length)

    
    
    
    def tokenize(self, text):
        sentences = tf.data.Dataset.from_tensor_slices(text)
        sequences = sentences.flat_map(self._tokenize)
        batch = sequences.padded_batch(1000000)
        return tf.data.experimental.get_single_element(batch)


    
    
    @tf.function(input_signature = [tf.TensorSpec(shape = [None], dtype = tf.string)])
    def __call__(self, text):    
        print('Tracing Translator!\n')
        outputs = self.beam_search._call(self.tokenize(text)) 

        sequences, probabilities = outputs['sequences'], outputs['probabilities']
        with tf.device('CPU'):  # without this device placement we get an error
            translations = tf.map_fn(self.outp_tokenizer.tf_detokenize, sequences, dtype=tf.string)
    
        return {'translations': translations, 'probabilities': probabilities}    