import tensorflow as tf



class Scale(tf.keras.layers.Layer):
    
    def __init__(self, factor, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.factor = factor
      
    
    
    def call(self, inputs):
        return inputs*self.factor
    
    
    
    def compute_mask(self, inputs, mask=None):
        return inputs._keras_mask
    
    
    
    def get_config(self):
        base_config = super(Scale, self).get_config()
        return {**base_config, 'factor': self.factor}
    


    
    
    
    
    
    
class MHA(tf.keras.layers.Layer):    # Multi Head Attention
    
    def __init__(self, d_query, d_value, n_heads, direction=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # adding some attributes
        self.d_query = d_query      # dimension of the query space
        self.d_value = d_value      # dimension of the value space
        self.n_heads = n_heads      # number of heads
        self.direction = direction  # time direction to which paying attention to
                                    # if not 'past' or 'future' is selected, it will be 'both'
                                    # default is 'None' resulting in 'both'
        
        # adding some dense layers
        self.dense_q = tf.keras.layers.Dense(d_query)  # for query
        self.dense_k = tf.keras.layers.Dense(d_query)  # for key 
        self.dense_v = tf.keras.layers.Dense(d_value)  # for value
        self.dense_a = tf.keras.layers.Dense(d_value)  # for attention
        
        
        self.c_query = self.d_query // self.n_heads   # num of query channels per head
        self.c_value = self.d_value // self.n_heads   # num of value channels per head
        
        
        
    def create_direction_mask(self, shape, dtype):
        """Create a boolean mask depending on the self.direction argument. This mask controls 
           whether or not information can flow forwards, backwards or in both time directions.
           
        Args:
            shape: shape == (2,), dtype == tf.int64   # shape of the boolean mask
            dtype: object of class tf.dtypesint64E  # the data type to which the mask shall be casted
            
        Returns:
            mask: shape == (2,), dtype = dtype
        """
        
        if self.direction == 'past':
            return tf.linalg.band_part(tf.ones(shape, dtype), -1, 0)
        elif self.direction == 'future':
            return tf.linalg.band_part(tf.ones(shape, dtype), 0, -1)
        else:
            return tf.ones(shape, dtype)
    
    
    
    def split_heads(self, x, channels):
        """Split the last dimension into (n_heads, channels).
           Transpose the result such that the shape is (batch_size, n_heads, n_query/key , channels).
           The last step allows broadcasting over n_heads."""
        
        x = tf.reshape(x, shape=(tf.shape(x)[0], -1, self.n_heads, channels))
        
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    
    
    def scaled_dot_product_attention(self, Query, Key, Value, Mask=None):
        """Calculate the dot product attention
    
        Args:
            Query: shape == (batch_size, n_heads, n_query, d_query)
            Key: shape == (batch_size, n_heads, n_key, d_query)
            Value: shape == (batch_size, n_heads, n_key, d_value)
            Mask: shape == (batch_size, n_heads, n_query, n_key) 
    
        Returns:
            attention_matrix: shape == (batch_size, n_heads, n_query, d_value)
            weight_matrix: shape == (batch_size, n_heads, n_query, n_key)
        """
        
        d_query = tf.cast(tf.shape(Query)[-1], dtype=Query.dtype)  # dimension of query as float

        score_matrix = tf.matmul(Query, Key, transpose_b=True) / tf.math.sqrt(d_query)   
        # (batch_size, n_heads, n_query, n_key)

        if Mask is not None:
            score_matrix += (1.0 - Mask) * -1e19 

        weight_matrix = tf.nn.softmax(score_matrix, axis = -1)   # (batch_size, n_heads, n_query, n_key)

        attention_matrix = tf.matmul(weight_matrix, Value)     # (batch_size, n_heads, n_query, d_value)
        
        return attention_matrix, weight_matrix
    
    
    
    def call(self, query, key, value):
        
        dtype = query.dtype  # the data type of query, key and value        
        batch_size = tf.shape(query)[0] # batch size of query, key and value 
             
        
        # To use tf.matmul(), it will be convenient to flatten the inner dimensions.
        # Ex: a tensor of shape [d0, d1, d2, d3, d4] will become a tensor of shape [d0, d1 + d2 + d3, d4]
        # This step will be reversed later.
        Query = tf.reshape(query, shape = [batch_size, -1, self.d_query] )  # flatten the inner dimensions
        Key = tf.reshape(key, shape = [batch_size, -1, self.d_query])       # flatten the inner dimensions 
        Value = tf.reshape(value, shape = [batch_size, -1, self.d_value])   # flatten the inner dimensions
        
        # computing the inner shapes, i.e. shapes without first (batch_size) and last dimension (d_query)
        # these shapes have been flattened in the previous step
        # Ex: in the previous example this is [d1, d2, d3]
        inner_query_shape = tf.shape(query)[1:-1]   
        inner_key_shape = tf.shape(key)[1:-1]
        
        # computing the size of the flattened inner shape, i.e. its total dimension
        # Ex: in the previous example this is d1 + d2 + d3
        n_query = tf.shape(Query)[1]  # number of queries
        n_key = tf.shape(Key)[1]      # number of keys
        
        # applying some initial dense layers
        Query = self.dense_q(Query)  # (batch_size, n_query, d_query)
        Key = self.dense_k(Key)      # (batch_size, n_key, d_query)
        Value = self.dense_v(Value)     # (batch_size, n_key, d_value)
        
        # splitting the last dimension into n_heads vectors and reordering the dimensions
        Query = self.split_heads(Query, self.c_query)  # (batch_size, n_heads, n_query, c_query)
        Key = self.split_heads(Key, self.c_query)      # (batch_size, n_heads, n_key, c_query)
        Value = self.split_heads(Value, self.c_value)  # (batch_size, n_heads, n_key, c_value)
        
        # create direction mask and add dimensions to allow broadcasting
        Mask = self.create_direction_mask([n_query, n_key], dtype)[tf.newaxis, tf.newaxis, : , : ]  
        
        # if a mask for 'value' is provided, it will be combined with the direction mask
        # for this, we cast it to dtype and add dimensions to allow broadcasting
        try:
            mask = tf.cast(value._keras_mask, dtype = dtype)[:, tf.newaxis, tf.newaxis, :]
            Mask *= mask
        except:
            #tf.print('No mask provided!')
            pass
        
        # compute the attention and the attention weights
        # attention shape = (batch_size, num_heads, n_query, c_value)
        # weights shape = (batch_size, num_heads, n_query, n_key)
        attention, weights = self.scaled_dot_product_attention(Query, Key, Value, Mask)
              
        # rearranging dimensions 
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])    # (batch_size, n_query, n_heads, c_value)

        # reshaping to reverse the flattening above
        # new attention shape = (batch_size, inner_query_shape, d_value) 
        # new weights shape = (batch_size, n_heads, inner_query_shape, inner_key_shape)
        new_attention_shape = tf.concat([[batch_size], inner_query_shape, [self.d_value]], axis=0)   
        attention = tf.reshape(attention, shape = new_attention_shape) 
        new_weights_shape = tf.concat([[batch_size, self.n_heads], inner_query_shape, inner_key_shape], axis=0)
        weights = tf.reshape(weights, shape = new_weights_shape)
        
        # applying the final dense layer
        attention = self.dense_a(attention)   # (batch_size, inner_query_shape, d_value)

        return attention, weights
      
        
    
    def compute_mask(self, query, mask=None):
        """Returns the Keras mask of the input query if available. As the input query is beeing processed 
           in the call() method, the Keras mask might have gone.
           
        Args:
            query: shape == (..., n_query, d_query), dtype == tf.float32
            mask: shape == (..., n_query, d_query),  dtype == tf.bool
            
        Returns:
            mask: shape == (..., n_query, d_query),  dtype == tf.bool
           """
        
        try:
            return query._keras_mask, None
        except:
            #tf.print('Query has no mask!')
            return None, None
            
       
    
    def get_config(self):
        base_config = super().get_config()
        config = {**base_config, 
                  "d_query": self.d_query,
                  "d_value": self.d_value, 
                  "n_heads": self.n_heads,
                  "c_query": self.c_query,
                  "c_value": self.c_value,
                  "direction": self.direction}
        return config
    
 

    @classmethod
    def from_config(cls, config):
        c_query = config.pop("c_query")
        c_value = config.pop("c_value")
        self = cls(**config)       
        self.c_query = c_query
        self.c_value = c_value
        return self


    
  


    
    

class MHAU(tf.keras.layers.Layer):   # Multi Head Attention Unit
    
    def __init__(self, d_query, d_value, d_hidden, n_heads, dropout_rate=0.1, direction=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_query = d_query
        self.d_value = d_value
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.direction = direction
        
        # layers for block 1
        self.MHA = MHA(d_query, d_value, n_heads, direction)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # layers for block 2
        self.dense1 = tf.keras.layers.Dense(d_hidden, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_value)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    
    
    def call(self, query, key, value, training):

        # block 1
        x, weights = self.MHA(query, key, value)
        x = self.dropout1(x, training = training)
        x = tf.keras.layers.Add()([x, query])
        x1 = self.layernorm1(x)

        # block 2
        x = self.dense1(x1)
        x = self.dense2(x)
        x = self.dropout2(x, training = training)
        x = tf.keras.layers.Add()([x, x1])
        x2 = self.layernorm2(x)

        return x2, weights
        
    
    
    def compute_mask(self, query, mask=None):
        try:
            return query._keras_mask, None
        except:
            #tf.print('Query has no mask!')
            return None, None
        
    
    
    def get_config(self):
        base_config = super().get_config()
        config = {**base_config,
                  "d_query": self.d_query,
                  "d_value": self.d_value,
                  "d_hidden": self.d_hidden,
                  "n_heads": self.n_heads,
                  "dropout_rate": self.dropout_rate,
                  "direction": self.direction}
        return config



    
    
    
    

class PositionalEmbedding(tf.keras.layers.Layer):
    
    def __init__(self, d_model, max_seq_length, vocab_size, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.factor = tf.math.sqrt(tf.cast(self.d_model, tf.float32))  
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True)
        self.pos_encoding = self.positional_encoding(max_seq_length=max_seq_length, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        
        
        
    def positional_encoding(self, max_seq_length, d_model):
        hdim = d_model // 2 # half-dimension
    
        exponents = tf.cast(tf.range(hdim)[tf.newaxis,:] / hdim, tf.float32)      # shape = [None, hdim], dtype = tf.float32
        positions = tf.cast(tf.range(max_seq_length)[:, tf.newaxis], tf.float32)  # shape = [seq_length, None], dtype = tf.float32
        angles = positions / tf.math.pow(10000., exponents)                       # shape = [seq_length, hdim]

        # use 'sin' for the first half and 'cos' for the second half of feature indices
        return tf.concat([tf.math.sin(angles), tf.math.cos(angles)], axis = 1)
    
    
    
    def call(self, x, training):
        x = self.embedding(x)
        x *= self.factor
        x += self.pos_encoding[ tf.newaxis , : tf.shape(x)[1], : ]
        x = self.dropout(x, training)
        return x
     
    
    
    def compute_mask(self, x, mask=None):
        return self.embedding.compute_mask(x)
        

    
    def get_config(self):
        base_config = super().get_config()
        config = {**base_config, 
                  "d_model": self.d_model,
                  "max_seq_length": self.max_seq_length,
                  "vocab_size": self.vocab_size,
                  "dropout_rate": self.dropout_rate,
                  "factor": self.factor.numpy(),
                  "pos_encoding": self.pos_encoding.numpy()}
        return config
    
    
    
    @classmethod
    def from_config(cls, config):
        factor = config.pop("factor")
        pos_encoding = config.pop("pos_encoding")
        self = cls(**config)       
        self.factor = tf.constant(factor, dtype=tf.float32)
        self.pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
        return self        
