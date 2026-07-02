import os
import time
import tensorflow as tf
from utils.layers import PositionalEmbedding, MHA, MHAU
from tensorflow.train import Int64List, Feature, Features, Example
import tensorflow_datasets as tfds


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
        
        # Note, we cut the <start> or <end> token later , hence the extra +1
        mask1 = tf.logical_and(tf.size(inp_sequence) <= max_inp_seq_length,
                              tf.size(outp_sequence) <= max_outp_seq_length + 1)  
        
        # at least one token different from <start> and <end>
        mask2 = tf.logical_and(tf.size(inp_sequence) > 0, tf.size(outp_sequence) > 2) 
       
        return tf.logical_and(mask1, mask2)
    
    
    
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



    


    
    
    
def load_tokenized_data(directory, 
                        file_prefix,
                        compression_type,
                        batch_size, 
                        max_inp_seq_length, 
                        max_outp_seq_length,
                        start_code, 
                        shuffle_buffer_size = 1, 
                        training = False, 
                        transformer = True):
    
    
    def cast_fn(tokens):
        inp_tokens = tf.cast(tf.sparse.to_dense(tokens['input_tokens']), tf.int32)
        outp_tokens = tf.cast(tf.sparse.to_dense(tokens['output_tokens']), tf.int32)
        
        # add a code for a start (code = vocab_size) and end (code = 0) token
        outp_tokens = tf.concat([[start_code] , outp_tokens , [0]], axis=0)
        
        return inp_tokens, outp_tokens


    
    
    def filter_max_length(inp_sequence, outp_sequence):
        """Creating a boolean mask marking all pairs of sequences of length <= max_length"""
        
        # Note, we cut the <start> or <end> token later , hence the extra +1
        mask1 = tf.logical_and(tf.size(inp_sequence) <= max_inp_seq_length,
                              tf.size(outp_sequence) <= max_outp_seq_length + 1)  
        
        # at least one token different from <start> and <end>
        mask2 = tf.logical_and(tf.size(inp_sequence) > 0, tf.size(outp_sequence) > 2) 
       
        return tf.logical_and(mask1, mask2)

    
    
    
    # create a list of all data file names
    file_list = sorted(tf.io.gfile.glob(os.path.join(directory, file_prefix+'*')))

    
    # load the file names into a dataset
    file_names = tf.data.Dataset.list_files(file_list, shuffle=False)
    serialized_seq = file_names.flat_map(lambda file_name: tf.data.TFRecordDataset(file_name, 
                                                            compression_type=compression_type))
    

    # define the features, needed for the next step
    features = {'input_tokens': tf.io.VarLenFeature(dtype=tf.int64),
               'output_tokens': tf.io.VarLenFeature(dtype=tf.int64),}
    
    
    # parse the serialized tokens into sequences of integers
    sequences = serialized_seq.map(lambda string: tf.io.parse_single_example(string, features), 
                                   num_parallel_calls=AUTOTUNE)
    sequences = sequences.map(cast_fn, num_parallel_calls=AUTOTUNE)

    
    # filter out too long sequences
    sequences = sequences.filter(filter_max_length).cache()
    
    
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
    
    
    
    
    
    
    
    
    
def serialize(inp_tokens, outp_tokens):
    inp_tokens = inp_tokens.numpy().flatten()
    outp_tokens = outp_tokens.numpy().flatten()
    
    feature = {'input_tokens':Feature(int64_list=Int64List(value=inp_tokens)),
              'output_tokens':Feature(int64_list=Int64List(value=outp_tokens))} 
    
    example = Example(features=Features(feature=feature))
    
    return tf.constant(example.SerializeToString(), tf.string)







def tf_serialize(inp_tokens, outp_tokens):
    return tf.py_function(serialize, inp=[inp_tokens, outp_tokens], Tout=tf.string)








def tokenize_text_and_save(text, 
                           tokenizer, 
                           directory, 
                           batch_size, 
                           compression_type = 'GZIP', # options: '', 'GZIP', 'ZLIB' 
                           print_time = False, 
                           num_files = None):
    
    
    # creating the target directory if it doesn't exist
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    
    # unpacking the tokenizers
    inp_tokenizer = tokenizer['input']
    outp_tokenizer = tokenizer['output']
    
    
    # loading the text corpus
    with open(text['input'], 'r') as f:
        inp_text = f.readlines()    
    with open(text['output'], 'r') as f:
        outp_text = f.readlines()
    
    
    # looping through the batches of data    
    start = time.time()    
    n = 1
    while len(inp_text) > 0 and len(outp_text) > 0:
        
        # tokenizing the input strings
        inp_batch = inp_text[:batch_size]        
        encoded_inp_batch = inp_tokenizer.tokenize_batch(inp_batch)
        inp_tokens = [encoded_line.ids for encoded_line in encoded_inp_batch]
        
        
        # tokenizing the output strings
        outp_batch = outp_text[:batch_size]
        encoded_outp_batch = outp_tokenizer.tokenize_batch(outp_batch)
        outp_tokens = [encoded_line.ids for encoded_line in encoded_outp_batch]
        
        
        # creating a token dataset and serializing the tokens
        tokens = (tf.ragged.constant(inp_tokens), tf.ragged.constant(outp_tokens))
        token_ds = tf.data.Dataset.from_tensor_slices(tokens)
        
        serialized_token_ds = token_ds.map(tf_serialize, num_parallel_calls=AUTOTUNE)
        
        
        # initializing the writer and writing the serialized tokens to disc
        file_name = os.path.join(directory, f'{file_prefix}-{n:04d}.tfrecord')
        writer = tf.data.experimental.TFRecordWriter(filename=file_name, 
                                                     compression_type=compression_type)
        writer.write(serialized_token_ds)
        
                    
        # removing the batch from the text corpus
        inp_text = inp_text[batch_size:]
        outp_text = outp_text[batch_size:]
        
        
        # reporting the computation time for this batch
        if print_time:
            print(f'Time for iteration {n}: {int(time.time()-start)} seconds.')
        start = time.time()

    
        # stopping if maximal number of files has been created
        if num_files != None and n == num_files:
            break        
        n += 1
    


    
    
    
    

def tokenize_dataset_and_save(dataset,
                              split,
                              tokenizer, 
                              directory,
                              file_prefix,
                              batch_size, 
                              compression_type = 'GZIP', # options: '', 'GZIP', 'ZLIB' 
                              print_time = False, 
                              num_files = None):
    
    
    # creating the target directory if it doesn't exist
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    
    # unpacking the tokenizers
    inp_tokenizer = tokenizer['input']
    outp_tokenizer = tokenizer['output']
    
    
    # loading the dataset and batching it
    text = tfds.load(name=dataset, split=split, with_info=False, as_supervised=True)
    batches = text.batch(batch_size).as_numpy_iterator()
    
    
    # looping through the batches of data
    start = time.time()    
    n = 1
    for inp_batch, outp_batch in batches:
                
        # tokenizing the input strings
        inp_batch = list(map(lambda s: s.decode(), inp_batch))        
        encoded_inp_batch = inp_tokenizer.tokenize_batch(inp_batch)
        inp_tokens = [encoded_line.ids for encoded_line in encoded_inp_batch]
        
        
        # tokenizing the output strings
        outp_batch = list(map(lambda s: s.decode(), outp_batch))
        encoded_outp_batch = outp_tokenizer.tokenize_batch(outp_batch)
        outp_tokens = [encoded_line.ids for encoded_line in encoded_outp_batch]
        
        
        # creating a token dataset and serializing the tokens
        tokens = (tf.ragged.constant(inp_tokens), tf.ragged.constant(outp_tokens))
        token_ds = tf.data.Dataset.from_tensor_slices(tokens)
        serialized_token_ds = token_ds.map(tf_serialize, num_parallel_calls=AUTOTUNE)
        
        
        # initializing the writer and writing the serialized tokens to disc
        file_name = os.path.join(directory, f'{file_prefix}-{n:04d}.tfrecord')
        writer = tf.data.experimental.TFRecordWriter(filename=file_name, 
                                                     compression_type=compression_type)
        writer.write(serialized_token_ds)
        
        
        # reporting the computation time for this batch
        if print_time:
            print(f'Time for iteration {n}: {int(time.time()-start)} seconds.')
        start = time.time()
    
    
        # stopping if maximal number of files has been created    
        if num_files != None and n == num_files:
            break        
        n += 1

        
        
        
        
        
        
def tokenize_dataset_and_save_v2( dataset,
                                  split,
                                  tokenizer, 
                                  directory,
                                  file_prefix,
                                  shard_size,
                                  batch_size, 
                                  max_inp_seq_length,
                                  max_outp_seq_length,
                                  compression_type = '', # options: '', 'GZIP', 'ZLIB' 
                                  print_time = False, 
                                  num_files = None):
    
    
    # creating the target directory if it doesn't exist
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    
    # unpacking the tokenizers
    inp_tokenizer = tokenizer['input']
    outp_tokenizer = tokenizer['output']
    
    
    
    def tokenize_batch(batch):
        inp_batch, outp_batch = batch
        
        # tokenizing the input strings
        inp_batch = list(map(lambda s: s.decode(), inp_batch.numpy()))        
        encoded_inp_batch = inp_tokenizer.tokenize_batch(inp_batch)
        inp_tokens = [encoded_line.ids for encoded_line in encoded_inp_batch]
        
        # tokenizing the output strings
        outp_batch = list(map(lambda s: s.decode(), outp_batch.numpy()))
        encoded_outp_batch = outp_tokenizer.tokenize_batch(outp_batch)
        outp_tokens = [encoded_line.ids for encoded_line in encoded_outp_batch]
        
        # filtering out too long sequences
        inp_tokens = [s for s in inp_tokens if (len(s) <= max_inp_seq_length and 0 < len(s))]
        outp_tokens = [s for s in outp_tokens if (len(s) <= max_outp_seq_length and 0 < len(s))]
        
        # padding and transforming into a tensor
        inp_tokens = tf.ragged.constant(inp_tokens, dtype=tf.int32).to_tensor()
        outp_tokens = tf.ragged.constant(outp_tokens, dtype=tf.int32).to_tensor()
        
        return inp_tokens, outp_tokens
    
    
    
    
    def tf_tokenize_batch(batch):
        return tf.py_function(tokenize_batch, inp=[batch], Tout=[tf.int32, tf.int32])
        
    
    
    def add_start_and_end(token_shard):
        inp_tokens, outp_tokens = token_shard
        shard_size = tf.shape(inp_shard)[0]
        
        # add a code for a start (code = vocab_size) and end (code = 0) token
        inp_tokens = tf.concat([tf.constant(inp_tokenizer.start_code, shape=[shard_size,1]), 
                                inp_tokens , 
                                tf.zeros(shape=[shard_size,1], dtype = tf.int32)], 
                               axis=1)
        # add a code for a start (code = vocab_size) and end (code = 0) token
        outp_tokens = tf.concat([tf.constant(outp_tokenizer.start_code, shape=[batch_size,1]), 
                                 outp_tokens , 
                                 tf.zeros(shape=[batch_size,1], dtype = tf.int32)], 
                                axis=1)
        
        return inp_tokens, outp_tokens
        
    
    
    
    def serialize_shard(shard):
        token_ds = tf.data.Dataset.from_tensor_slices(shard).batch(batch_size)
        serialized_token_ds = token_ds.map(tf_serialize, num_parallel_calls=AUTOTUNE)
        return serialized_token_ds
    
    
    
    
    
    
    # loading the dataset and splitting it into shards (large batches)
    text = tfds.load(name=dataset, split=split, with_info=False, as_supervised=True)
    text_batches = text.batch(shard_size)
    
        
    # tokenizing the data and add start code
    token_shards = text_shards.map(tf_tokenize_batch, num_parallel_calls=AUTOTUNE)
    token_shards = token_shards.map(add_start_and_end, num_parallel_calls=AUTOTUNE)
    
    # serializing the tokens
    serialized_token_shards = token_shards.apply(serialize_shard)    
    
    
    # writing the serialized tokens into a files, one per shard
    start = time.time()
    n = 1
    for shard in serialized_token_shards:                
        # initializing the writer and writing the serialized tokens to disc
        file_name = os.path.join(directory, f'{file_prefix}-{n:04d}.tfrecord')
        writer = tf.data.experimental.TFRecordWriter(filename=file_name, 
                                                     compression_type=compression_type)
        writer.write(shard)
        
        
        # reporting the computation time for this batch
        if print_time:
            print(f'Time for iteration {n}: {int(time.time()-start)} seconds.')
        start = time.time()
    
    
        # stopping if maximal number of files has been created    
        if num_files != None and n == num_files:
            break        
        n += 1
    
        
        
        
        
        
        
        
        
        
        
def input_fn_v2(dataset,
                tokenizer, 
                batch_size,  
                shard_size, # e.g. 1000*batch_size
                max_inp_seq_length,
                max_outp_seq_length,
                shuffle_buffer_size = 1, 
                training = False, 
                transformer = True,
                swap = False
                ):
    

    
    # unpacking the tokenizers
    inp_tokenizer = tokenizer['input']
    outp_tokenizer = tokenizer['output']
    
    

    
    
    
    def tokenize_batch(inp_batch, outp_batch):
        
        # tokenizing the input strings and filtering out too long sequences
        inp_batch = list(map(lambda s: s.decode(), inp_batch.numpy()))        
        inp_enc = inp_tokenizer.tokenize_batch(inp_batch)
        
        
        # tokenizing the output strings and filtering out too long sequences
        outp_batch = list(map(lambda s: s.decode(), outp_batch.numpy()))
        outp_enc = outp_tokenizer.tokenize_batch(outp_batch)
        
        
        tokens = [(i.ids, o.ids) for (i,o) in zip(inp_enc, outp_enc) if len(i) <= max_inp_seq_length and 0 < len(i) and len(o) < max_outp_seq_length and 0 < len(o)]
        
        
        # padding and transforming into a tensor, we add a zero token at the end
        return tf.ragged.constant(tokens, dtype=tf.int32).to_tensor(
                        shape=[None, max_inp_seq_length, max_outp_seq_length])
        

        
    
    def tf_tokenize_batch(inp_batch, outp_batch):
        return tf.py_function(tokenize_batch, inp=[inp_batch, outp_batch], Tout=tf.int32)
    
    
    
    def form_batches(tokens):
        batch_length = tf.shape(tokens)[0]

        # we add a start token
        return ({'encoder_input': tokens[:,0,:], 
                'decoder_input': tf.pad(tokens[:,1,:-1], 
                                        paddings=tf.constant([[0,0],[1,0]]),                                                         constant_values=outp_tokenizer.vocab_size)},
                 tokens[:,1,:])
        
    
   
    # shuffling the data
    if training:
        dataset = dataset.shuffle(shuffle_buffer_size, seed=1)
    
    
    # batching the shard
    shards = dataset.batch(shard_size)  
    
    
    #swapping input and output
    if swap:
        shards = shards.map(lambda x, y: (y,x), num_parallel_calls=AUTOTUNE)
    
    
    # tokenizing the data 
    shards = shards.map(tf_tokenize_batch, num_parallel_calls=AUTOTUNE)
    shards = shards.map(form_batches, num_parallel_calls=AUTOTUNE)
    
    
    # resize the batches
    batches = shards.unbatch().batch(batch_size)
    
    
    if training:
        batches = batches.repeat(None)
    
    
    return batches.prefetch(AUTOTUNE)