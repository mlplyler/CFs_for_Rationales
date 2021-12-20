import tensorflow as tf
import numpy as np
import sys
import random
sys.path.append('../share/')
from ratstuff import *
from mybert import *


@tf.function         
def cag_js2s_single(args,cfmodel,x_in,x_tar,bsize,
                            opt_cf,train=True):  
  '''                            
  cfmodel train for a single model
  '''                 
  with tf.GradientTape() as cf_tape:
    logits,dec_ids,masks,cost_d = cfmodel.call_train(x_in,tar=x_tar,bsize=bsize,
                                  train=train,force=True,
                                  from_logits=False)  
    cfvars = cfmodel.trainable_variables
    gradientscf = cf_tape.gradient(cost_d,cfvars)
    opt_cf.apply_gradients(zip(gradientscf,cfvars))
  return(dec_ids,cost_d)       

  
@tf.function         
def jpredict_js2s_single(args,cfmodel,x_in,x_tar,bsize,
                            opt_cf,train=True,t_out=None):  
  '''                            
  cfmodel train for a single model
  '''                 
  logits,dec_ids,masks,cost_d = cfmodel.call_eval(x_in,tar=x_tar,bsize=bsize,
                                  train=train,force=True,
                                  from_logits=False,t_out=t_out)  
  return(dec_ids,cost_d)           
      
@tf.function         
def cag_js2s_single_wmask(args,cfmodel,x_in,x_tar,mybmask,bsize,
                            opt_cf,train=True):  
  '''                            
  cfmodel train for a single model
  '''                 
  with tf.GradientTape() as cf_tape:
    logits,dec_ids,masks,cost_d = cfmodel.call_train(x_in,tar=x_tar,bsize=bsize,
                                  train=train,force=True,
                                  from_logits=False,mybmask=mybmask)  
    cfvars = cfmodel.trainable_variables
    gradientscf = cf_tape.gradient(cost_d,cfvars)
    opt_cf.apply_gradients(zip(gradientscf,cfvars))
  return(dec_ids,cost_d) 

#@tf.function         
def cag_js2s_single_wmask_L2(args,cfmodel,x_in,x_tar,mybmask,bsize,
                            opt_cf,train=True):  
  '''                            
  cfmodel train for a single model
  '''                 
  with tf.GradientTape() as cf_tape:
    logits,dec_ids,masks,cost_d = cfmodel.call_train(x_in,tar=x_tar,bsize=bsize,
                                  train=train,force=True,
                                  from_logits=False,mybmask=mybmask)                                   
    theloss = cost_d + cfmodel.getL2loss()    
    cfvars = cfmodel.trainable_variables
    gradientscf = cf_tape.gradient(theloss,cfvars)
    opt_cf.apply_gradients(zip(gradientscf,cfvars))
  return(dec_ids,cost_d)       
        

  
@tf.function         
def jpredict_js2s_single_wmask(args,cfmodel,x_in,x_tar,mybmask,bsize,
                            opt_cf,train=False,t_out=None):  
  '''                            
  cfmodel train for a single model
  '''                 
  logits,dec_ids,masks,cost_d = cfmodel.call_eval(x_in,tar=x_tar,bsize=bsize,
                                  train=train,force=True,
                                  from_logits=False,t_out=t_out,mybmask=mybmask)  
  return(dec_ids,cost_d)           
      

  
  

def compute_loss_dec(logits,x_masked,xtrue,pad_id,one_tok,zero_tok,mask=None):  
  loss = tf.keras.losses.sparse_categorical_crossentropy(
                        y_true=xtrue,y_pred=logits+1e-10, 
                        from_logits=True)
  if mask is None:
    mask = tf.cast(tf.math.equal(x_masked, one_tok),dtype=tf.float32)+\
          tf.cast(tf.math.equal(x_masked, zero_tok),dtype=tf.float32)
  
  loss = loss*mask
                                                   
  loss = tf.reduce_sum(loss,axis=-1)
  loss = tf.reduce_mean(loss,axis=-1)                                                   
  return(loss)        
######################################################3
class myModel(tf.keras.Model):
  def __init__(self,args,embedding_layer,
               generator=None,encoder=None,outlayer=None):
    super(myModel,self).__init__()

    if 'cf_dropout' not in args:
      args['cf_dropout']=.1
    if 'cf_mpe' not in args:
      args['cf_mpe']=args['mpe']
    self.args = args    
    self.berty = myBert(num_layers=args['cf_nlayers'],
                            d_model=args['cf_dmodel'],
                            num_heads = args['cf_heads'],
                            dff=args['cf_dff'],
                            input_vocab_size=args['n_v'],
                            pe_input=args['cf_mpe'],
                            rate=args['cf_dropout']
                            )
    self.padding_id = tf.convert_to_tensor(embedding_layer.vocab_map["<padding>"],
                                        dtype=tf.int32)
    self.start_id = tf.convert_to_tensor(embedding_layer.vocab_map["<start>"],
                                        dtype=tf.int32)
    self.onetok = tf.convert_to_tensor(embedding_layer.vocab_map["<one>"],
                                        dtype=tf.int32)
    self.zerotok = tf.convert_to_tensor(embedding_layer.vocab_map["<zero>"],
                                        dtype=tf.int32)  
    self.end_id = tf.convert_to_tensor(embedding_layer.vocab_map["<end>"],
                                        dtype=tf.int32)  


  @tf.function(experimental_relax_shapes=True)
  def call(self,x,tar,bsize,train,force=False,
              from_logits=False,temper=1e-6,dec_ids=None,truex=None,t_out=None,
              mybmask=None):              
    ep_mask = create_padding_mask(x,self.padding_id)
    dlogits = self.berty(x, 
                         training=train, 
                         enc_padding_mask=ep_mask)
    return(dlogits)

  @tf.function(experimental_relax_shapes=True)
  def call_train(self,x,tar,bsize,train,force=False,
              from_logits=False,temper=1e-6,dec_ids=None,truex=None,t_out=None,
              mybmask=None):              
    ep_mask = create_padding_mask(x,self.padding_id)    
    dlogits = self.berty(x, 
                         training=train, 
                         enc_padding_mask=ep_mask)
    masks = 1-tf.cast(tf.math.equal(x, self.padding_id),dtype=tf.float32)
    dec_ids = dlogits[:,:,0]
    loss = compute_loss_dec(logits=dlogits,
                            x_masked=x,
                            xtrue=tar,
                            pad_id=self.padding_id,
                            one_tok=self.onetok,
                            zero_tok=self.zerotok,
                            mask=mybmask)
    return(dlogits,dec_ids,masks,loss)
    
  @tf.function(experimental_relax_shapes=True)
  def call_eval(self,x,tar,bsize,train,force=False,
              from_logits=False,temper=1e-6,dec_ids=None,truex=None,t_out=None,
              mybmask=None):              
    ep_mask = create_padding_mask(x,self.padding_id)
    dlogits = self.berty(x, 
                         training=train, 
                         enc_padding_mask=ep_mask)
    masks = 1-tf.cast(tf.math.equal(x, self.padding_id),dtype=tf.float32)
    dec_ids = tf.argmax(dlogits,axis=-1)
    loss = compute_loss_dec(logits=dlogits,
                            x_masked=x,
                            xtrue=tar,
                            pad_id=self.padding_id,
                            one_tok=self.onetok,
                            zero_tok=self.zerotok,
                            mask=mybmask)
    return(dlogits,dec_ids,masks,loss)


  @tf.function(experimental_relax_shapes=True)
  def call_eval_nounk(self,x,tar,bsize,train,force=False,
              from_logits=False,temper=1e-6,dec_ids=None,truex=None,t_out=None,
              mybmask=None,unkid=0):
    ep_mask = create_padding_mask(x,self.padding_id)
    dlogits = self.berty(x, 
                         training=train, 
                         enc_padding_mask=ep_mask)
    ## there are 7 <> tokens....
    masks = 1-tf.cast(tf.math.equal(x, self.padding_id),dtype=tf.float32)

    unkmask = tf.one_hot(
                indices=tf.ones_like(x)*unkid,
                depth = self.args['n_v']
    )
    dlogits = dlogits - 1e10*unkmask 
    dec_ids = tf.argmax(dlogits,axis=-1)
    loss = compute_loss_dec(logits=dlogits,
                            x_masked=x,
                            xtrue=tar,
                            pad_id=self.padding_id,
                            one_tok=self.onetok,
                            zero_tok=self.zerotok,
                            mask=mybmask)
    return(dlogits,dec_ids,masks,loss)    
  def getL2loss(self,):
      lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in self.trainable_variables
                   if 'bias' not in v.name.lower() ]) * self.args['l2_reg']
      return(lossL2)
    
    
         
def add_default_args(args,rollrandom=True):
  defaultargs = {
  "warmup":10000,
  "peak_lr":1e-4,
  "l2_reg":0.0,
  "log_path":"",
  "load_chkpt_dir":"",
  "train_file":"",
  "dev_file":"",
  "test_file":"",
  "embedding" :"",
  "embdim":100,
  "fixembs":False,
  "hidden_dimension":100,
  
  "tnlayers":4,
  "hidden_dimension":128,
  "tdff":512,
  "theads":8,  
  
  "cf_nlayers":4,
  "cf_dmodel":128,
  "cf_heads":8,
  "cf_dff":512,
  "max_pos_enc":10000,
  
  "numclass":1,
  "binarize":1,
  "evenup":1,
  "aspects" : [0,1,2,3,4],
  "split_rule":"six",
  "max_len" :256 ,
  "train_batch":64,
  "eval_batch":64,
  "gen_lr":1e-4,
  "enc_lr":1e-4,
  "cf_lr":1e-4,
  "gtype":"gru",
  "etype":"gru",
  "slevel":0.10,
  "wanted_sparse":0.10,
  "sparse_margin":.05,
  "vocentlam":1.0,
  "sparsity":1.0,
  "coherent":1.0,
  "costz01":1.0,
  "costz02":1.0,
  "costz12":1.0,
  "scost":"L2",
  "outact":"softmax",
  "trate":5e-4,
  "dropout":0.0,
  "lrdecay":0.99,
  "dosave":1,
  "reload":0,
  "checknum":2000,
  "start_epoch":0,
  "epochs_jrat":10,
  "epochs_js2s":30,
  "epochs_jcf":30,
  "max_epochs":30,
  "min_epochs":5,
  "beta1":0.9,
  "beta2":0.999,
  "initialization":"rand_uni",
  "edump":0,
  "mtype":"cfbert",
  "masktype":"cont",
  "classpick":-1,
  "hardzero":0,  
  "zact":False,
  "cf_lambda":1e-3, 
  "n_v":2**15, 
  "logfile" : "logfile.log"}
  newargs = dict(defaultargs)
  theseed = random.randint(1,10000000)
  newargs['rand_seed']=theseed ## this will be overwritten if in args

  for k in args:
    newargs[k]=args[k]
  k=0
  for i in range(len(newargs['aspects'])):
    for j in range(i+1,len(newargs['aspects'])):
      k+=1
  if k==0:
    k+=1      
  newargs['oversize']=k    
  if newargs['oversize']==0:
    newargs['oversize']=1
  
  return(newargs)      
        


