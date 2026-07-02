
import tensorflow as tf
from utils.layers import PositionalEmbedding, MHA, MHAU


AUTOTUNE = tf.data.experimental.AUTOTUNE


    
    
    











def build_transformer(d_model, d_hidden, n_heads, n_enc_layers, n_dec_layers, dropout_rate, 
                      max_inp_seq_length, max_outp_seq_length, inp_set_size, outp_set_size, epsilon = 1e-6):
    
    # Encoder block
    input_enc = tf.keras.layers.Input(shape=[None], dtype=tf.int32, name='encoder_input')
    x_enc = PositionalEmbedding(d_model, max_inp_seq_length, inp_set_size, dropout_rate)(input_enc)
    
    for _ in range(n_enc_layers):
        x_enc, _ = MHAU(d_query=d_model, d_value=d_model, d_hidden=d_hidden, 
                        n_heads=n_heads, dropout_rate=dropout_rate, direction=None 
                        )(query=x_enc, key=x_enc, value=x_enc)


        
    # Decoder block
    input_dec = tf.keras.layers.Input(shape=[None], dtype=tf.int32, name='decoder_input')
    x_dec = PositionalEmbedding(d_model, max_outp_seq_length, outp_set_size, dropout_rate)(input_dec)
    
    for _ in range(n_dec_layers):
        x_tmp, _ = MHA(d_query=d_model, d_value=d_model, n_heads=n_heads, 
                       direction='past')(query=x_dec, key=x_dec, value=x_dec)
        x_tmp = tf.keras.layers.Dropout(rate=dropout_rate)(x_tmp)
        x_tmp = tf.keras.layers.Add()([x_tmp, x_dec])
        x_dec = tf.keras.layers.LayerNormalization(epsilon=epsilon)(x_tmp)
        x_dec, _ = MHAU(d_query=d_model, d_value=d_model, d_hidden=d_hidden, 
                        n_heads=n_heads, dropout_rate=dropout_rate, direction=None 
                        )(query=x_dec, key=x_enc, value=x_enc)
    
    output_dec = tf.keras.layers.Dense(units=outp_set_size - 1 , name='decoder_output')(x_dec)  # no unit for the <start> code



    return tf.keras.models.Model(inputs={'encoder_input': input_enc, 'decoder_input': input_dec}, outputs=output_dec)
                   


    
    
    
        






def build_combined_model(model):
    inputs = {name: inp for (name, inp) in zip(model.input_names, model.inputs)}

    ckpts = [os.path.join(model.checkpoint_path, f"{epoch:03d}-{metric:.3f}") for (epoch, metric) in model.top_epochs]
    custom_objects = {'MHA': MHA, 
                      'MHAU': MHAU, 
                      'PositionalEmbedding': PositionalEmbedding, 
                      'Weighted_Loss': Weighted_Loss,
                      'Factor': Factor,
                      'CustomSchedule': CustomSchedule,
                      'My_Callback': My_Callback}
    models = [ tf.keras.models.load_model(ckpt, custom_objects=custom_objects) for ckpt in ckpts]
    
    # renaming the models to avoid conflicts while initializing the combined model later
    for i, _model in enumerate(models):
        _model._name = model.name + f'_{i}'
    
    # computing the mean of all outputs by adding them up and dividing through the number of models
    outp_sum = tf.keras.layers.Add()([_model(inputs) for _model in models])
    factor = tf.constant(1./len(ckpts), outp_sum.dtype)
    outp = Scale(factor)(outp_sum)
    
    # initializing and compiling the combined model
    combined_model = tf.keras.Model(inputs=inputs, outputs = outp)
    combined_model.compile( loss = model.loss,
                            optimizer =  model.optimizer,
                            metrics = model.compiled_metrics._metrics,
                            weighted_metrics = model.compiled_metrics._weighted_metrics)

    return combined_model









class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmups=4000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.warmups = warmups
    
    def __call__(self, step):
        lrate = self.d_model**-0.5 * tf.minimum(step**-0.5, step*self.warmups**-1.5)
        return lrate
    
    def get_config(self):        
        config = {'d_model': self.d_model, 'warmups': self.warmups}
        return config
    
    

    
    
    
    
    
    
    

def input_fn(corpus, tokenizer, max_inp_seq_length, max_outp_seq_length, batch_size = 64, shuffle_buffer_size = 1, 
             training=False, transformer=True):
    """Takes a corpus dataset consisting of pairs. Each pair is a sentence in the input language and its 
       translation into the output language. Both sentences will be tokenized 
       resulting in a pair of sequences of integers. The dataset of such pairs will be batched,
       padded and prefetched.
       
    Args:
        corpus, a dataset of text: shape == (2,), dtype == tf.string
        tokenizer, a dictionary of SubwordTextEncoders, keys = ['input', 'output']
        max_length: shape == (), dtype == tf.int64
        batch_size: shape == (), dtype == tf.int64
        shuffle_buffer_size: shape == (), dtype == tf.int64
        training: shape == (), dtype == tf.bool   
        
        
    Returns:
        dataset of batches: shape == (2, batch_size, None), dtype == tf.int64
    """
    
    inp_tokenizer = tokenizer['input']
    outp_tokenizer = tokenizer['output']
   
        
        
    def tokenize(inp_sentence, outp_sentence):
        """Tokenizing and encoding our input and output sentences"""
        
        inp_sequence = inp_tokenizer.tf_tokenize(inp_sentence)
        outp_sequence = outp_tokenizer.tf_tokenize(outp_sentence)
        
        # add a code for a start (code = vocab_size) and end (code = 0) token
        outp_sequence = tf.concat([[outp_tokenizer.start_code] , outp_sequence , [0]], axis=0)
        
        return inp_sequence, outp_sequence
            


    
    def filter_max_length(inp_sequence, outp_sequence):
        """Creating a boolean mask marking all pairs of sequences of length <= max_length"""
        
        mask = tf.logical_and(tf.size(inp_sequence) <= max_inp_seq_length,
                              tf.size(outp_sequence) <= max_outp_seq_length + 1)  # we cut the <start> or <end> token later 
        
        return mask
    
    
    
    # apply the tokenize and filter functions; cache, shuffle and pad the dataset
    sequences = corpus.map(tokenize, num_parallel_calls=AUTOTUNE).filter(filter_max_length).cache()
    
    # if in training mode, shuffle the data and repeat infinitely many times
    if training:
        sequences = sequences.shuffle(shuffle_buffer_size, seed=1).repeat(None)
    
    # batch the sequences and apply padding, sequence length will be the longest in a batch
    batches = sequences.padded_batch(batch_size)
                
    # if the dataset feeds the transformer, rearrange input and output
    if transformer:
        batches = batches.map(lambda x, y: ({ 'encoder_input': x, 'decoder_input': y[:,:-1] }, y[:,1:]), 
                              num_parallel_calls=AUTOTUNE)

        
    return batches.prefetch(AUTOTUNE)



    
    
    
    

    
    
def show_example_batch(dataset):
    x, y = next(iter(dataset))
    print("An example bath: \n")
    print("encoder input:")
    print(x['encoder_input'])
    print()
    print("decoder input:")
    print(x['decoder_input'])
    print()
    print("true decoder output:")
    print(y)
    print()