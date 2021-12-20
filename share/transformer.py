
import tensorflow as tf


import time
import numpy as np


class Embedding_wlogits(tf.keras.layers.Layer):
  def __init__(self,target_vocab_size,d_model):
    super(Embedding_wlogits, self).__init__()
    self.embedding = tf.keras.layers.Embedding(target_vocab_size,d_model)
  def call(self,x,from_logits=False,temper = 1e-5):
    if not from_logits: 
      return(self.embedding(x))
    else:
      return(self.call_from_logits(x,temper))
    return(out)
  def call_from_logits(self,x,temper):    
    x_hard = get_max_mask(x)    
    x = tf.stop_gradient(x_hard-x)+x    
    out = tf.linalg.matmul(x,self.embedding.embeddings) 
    return(out)
def get_max_mask(arr,K=1):
  '''
  magic from
  https://stackoverflow.com/questions/43294421/
  
  returns a binary array of shape array 
  where the 1s are at the topK values along axis -1
  '''
  values, indices = tf.nn.top_k(arr, k=K, sorted=False)
  temp_indices = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(
        tf.shape(arr)[:(arr.get_shape().ndims - 1)]) + [K])], indexing='ij')
  temp_indices = tf.stack(temp_indices[:-1] + [indices], axis=-1)
  full_indices = tf.reshape(temp_indices, [-1, arr.get_shape().ndims])
  values = tf.reshape(values, [-1])
  
  mask_st = tf.SparseTensor(indices=tf.cast(
        full_indices, dtype=tf.int64), values=tf.ones_like(values), dense_shape=arr.shape)

  mask = tf.sparse.to_dense(tf.sparse.reorder(mask_st),default_value=0)  
  return(mask)  



## modified from
## https://www.tensorflow.org/tutorials/text/transformer  ,
#@title Transformer code from tutorial
def encode(lang1, lang2):
  lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
      lang1.numpy()) + [tokenizer_pt.vocab_size+1]

  lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
      lang2.numpy()) + [tokenizer_en.vocab_size+1]
  
  return lang1, lang2

def tf_encode(pt, en):
  result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
  result_pt.set_shape([None])
  result_en.set_shape([None])

  return result_pt, result_en

def filter_max_length(x, y, max_length=256):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates  
def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)  
def create_padding_mask(seq,padid):
  seq = tf.cast(tf.math.equal(seq, padid), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)  

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  #return mask  # (seq_len, seq_len)  
  return tf.cast(mask,dtype=tf.float32)  # (seq_len, seq_len)    
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk) 
                   

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  ###########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights  
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights 

class point_wise_feed_forward_network(tf.keras.layers.Layer):
  def __init__(self,d_model,dff):
    super(point_wise_feed_forward_network,self).__init__()
    self.d1 = tf.keras.layers.Dense(dff,activation='relu')# (batch_size, seq_len, dff)
    self.d2 = tf.keras.layers.Dense(d_model) # (batch_size, seq_len, d_model)
  def call(self,x):
    x = self.d1(x)
    x = self.d2(x)
    return(x)
    
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
  #@tf.function  
  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    
    #print('called ffn!!')
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2  
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    #print('decoder layer', np.shape(x), np.shape(look_ahead_mask))
    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    
    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    return out3
    
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1,padding_id=None):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.padding_id = padding_id
    
    
    self.embedding = Embedding_wlogits(input_vocab_size, d_model)    
    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)

  def call_z(self, x, training, mask,z,from_logits=False,temper=1e-5):

    seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.
    
    x = self.embedding(x,from_logits=from_logits,temper=temper)  # (batch_size, input_seq_len, d_model)
    
    ##########
    x = x*z
    ###########
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  # (batch_size, input_seq_len, d_model) 
  def call_z2(self, x, training, mask,z):
  
    '''
    shift the nonmasked stuff left and add padding
    '''  

    seq_len = tf.shape(x)[1]

    x = tf.concat([tf.expand_dims(x,axis=-1),
                    tf.cast(z,tf.int32)],
                  axis=-1)
    x = tf.map_fn(self.split_and_merge,x)
    
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  # (batch_size, input_seq_len, d_model) 
  def split_and_merge(self,vals_mask):#(values,mask,padding_id=69):
    out0,out1= tf.dynamic_partition(vals_mask[:,0],tf.cast(vals_mask[:,1],dtype=tf.int32),2)
    out = tf.concat([out1,
              (out0*0)+self.padding_id],axis=-1)
    return(out)  
       
  #@tf.function      
  def call(self, x, training, mask,from_logits=False,temper=1e-5):

    seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.    
    x = self.embedding(x,from_logits,temper)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]    
    
    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  # (batch_size, input_seq_len, d_model) 
    

    
    
    
class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = Embedding_wlogits(target_vocab_size,d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask,from_logits=False,temper=1e-5):

    seq_len = tf.shape(x)[1]
    
    x = self.embedding(x,from_logits,temper)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      
      x = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
            
    return x
    
  def call_wstate(self,x,state,enc_output,training,look_ahead_mask,padding_mask,
                   from_logits=False,temper=1e-5):
    '''
    x is just one token (B,1)
    https://github.com/tag-and-generate/tagger-generator/blob/master/tag-and-generate-train/src/transformer.py
    '''
    if state is not None:
      seq_len = tf.shape(state)[1]+1
    else:
      seq_len = tf.shape(x)[1]
   
    x = self.embedding(x,from_logits,temper)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    
    x += self.pos_encoding[:, seq_len, :]
    if state is not None:
      x = tf.concat([state,x],axis=1)
    state=x
    
    
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):    
      x = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
      
    return x,state
                   
class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
  def call(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask,from_logits=False,temper=1e-5):


    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
    dec_output = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask,
        from_logits,temper)
    
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output#, attention_weights  
  def call_enc(self,inp,training,enc_padding_mask):
    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    return(enc_output)

    
  ##@tf.function  
  def call_dec(self,enc_output,xtok,state,training,look_ahead_mask,dec_padding_mask,
                from_logits=False,temper=1e-5):
    #x,state,enc_output,            
    dec_output,state = self.decoder.call_wstate(
        xtok,state, enc_output, training, look_ahead_mask, dec_padding_mask,
        from_logits,temper)
    
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    return(final_output,state)#,attention_weights)


         
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)      

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)    
def create_masks(inp, tar,padid_inp,padid_tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp,padid_inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp,padid_inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar,padid_tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  #print('CREATE MASKS', np.shape(enc_padding_mask),np.shape(combined_mask),np.shape(dec_padding_mask))
  ##CREATE MASKS (25, 1, 1, 256) (25, 1, 258, 258) (25, 1, 1, 256)

  return enc_padding_mask, combined_mask, dec_padding_mask  
  
def create_masks_dec(dec_padding_mask, tar,padid_inp,padid_tar):
  # Encoder padding mask
  #enc_padding_mask = create_padding_mask(inp,padid_inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  #dec_padding_mask = create_padding_mask(inp,padid_inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar,padid_tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  #print('CREATE MASKS', np.shape(enc_padding_mask),np.shape(combined_mask),np.shape(dec_padding_mask))
  ##CREATE MASKS (25, 1, 1, 256) (25, 1, 258, 258) (25, 1, 1, 256)

  return combined_mask#, dec_padding_mask  
  
def create_masks_dec2(dec_padding_mask, tarshape,padid_inp,padid_tar):
  # Encoder padding mask
  #enc_padding_mask = create_padding_mask(inp,padid_inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  #dec_padding_mask = create_padding_mask(inp,padid_inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tarshape[1])
  dec_target_padding_mask = tf.ones(tarshape,dtype=tf.float32)
  dec_target_padding_mask = dec_target_padding_mask[:, tf.newaxis, tf.newaxis, :] 
  ###dec_target_padding_mask = create_padding_mask(tarshape,padid_tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  #print('CREATE MASKS', np.shape(enc_padding_mask),np.shape(combined_mask),np.shape(dec_padding_mask))
  ##CREATE MASKS (25, 1, 1, 256) (25, 1, 258, 258) (25, 1, 1, 256)

   
def create_masks_for_logits(inp, tar,padid_inp,padid_tar):
  # Encoder padding mask
  #enc_padding_mask = create_padding_mask(inp,padid_inp)
  inpshape=tf.shape(inp)
  enc_padding_mask = tf.zeros(shape=(inpshape[0],1,1,inpshape[1]),dtype=tf.float32)
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  #dec_padding_mask = create_padding_mask(inp,padid_inp)
  dec_padding_mask = tf.zeros(shape=(inpshape[0],1,1,inpshape[1]),dtype=tf.float32)
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = tf.zeros(shape=(inpshape[0],1,1,tf.shape(tar)[1]),dtype=tf.float32)
  #dec_target_padding_mask = create_padding_mask(tar,padid_tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  ####combined_mask = look_ahead_mask
  #print('CREATE MASKS FOR LOGITS', np.shape(enc_padding_mask),np.shape(combined_mask),np.shape(dec_padding_mask))
  return enc_padding_mask, combined_mask, dec_padding_mask   
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
   
