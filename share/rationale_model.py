import tensorflow as tf
import numpy as np
import sys
import random
from ratstuff import *
from mrat_utils import *
from transformer import Encoder as Tran_Encoder
from transformer import create_padding_mask




@tf.function
def compute_apply_gradients(args,model,x,y,
                optimizer_gen0,optimizer_enc0,
                train,bsize): 
                
  '''
  we only use optimizer_enc0 but they others are passed for convenience
  '''
  print('args sparsity', args['sparsity'])
  with tf.GradientTape() as gen_tape0, tf.GradientTape() as enc_tape0:
    allpreds,allzs,allzsum,allzdiff,allpkept,padmask = model(x,
                                                  train=train,bsize=bsize)    
    allcostgs,allcostes,alllosss,allobjs,allcostovers,allsparse,allcoherent = get_loss_sc(
                                          args,y,allpreds,
                                          allzs,allzsum,allzdiff,allpkept,padmask)                                                  
    all_e_cost = tf.reduce_sum(allcostes)
    all_g_cost = tf.reduce_sum(allcostgs)
                  
  ####################
  evars=model.encoders.trainable_variables + \
        model.outlayers.trainable_variables 
  

  gradientse = enc_tape0.gradient(all_e_cost, evars)  
  optimizer_enc0.apply_gradients(zip(gradientse, 
                                    evars))

  ####################
  ## apply generator gradients
  gvars = model.generator.trainable_variables

  gradientsg = gen_tape0.gradient(all_g_cost, gvars)
  optimizer_gen0.apply_gradients(zip(gradientsg, 
                                    gvars))


  allflex = [0]
  return(allcostgs,allcostes,allzsum,allzdiff,allpkept,allcostovers,
          allobjs,alllosss,allsparse,allcoherent,allflex)


@tf.function#(input_signamture=[])
def compute_apply_gradients_nogentrain(args,model,x,y,
                optimizer_gen0,optimizer_enc0,
                train,bsize): 
  print('args sparsity', args['sparsity'])
  with tf.GradientTape() as enc_tape0:
    allpreds,allzs,allzsum,allzdiff,allpkept,padmask = model(x,train=train,bsize=bsize)
    allcostgs,allcostes,alllosss,allobjs,allcostovers,allsparse,allcoherent = get_loss_sc(
                                                  args,y,allpreds,
                                                  allzs,allzsum,allzdiff,allpkept,padmask)                                                
    all_e_cost = tf.reduce_sum(allcostes)
    all_g_cost = tf.reduce_sum(allcostgs)
  ####################
  evars=model.encoders.trainable_variables + \
        model.outlayers.trainable_variables 
  

  gradientse = enc_tape0.gradient(all_e_cost, evars)  
  optimizer_enc0.apply_gradients(zip(gradientse, 
                                    evars))

  allflex = [0]
  return(allcostgs,allcostes,allzsum,allzdiff,allpkept,allcostovers,
          allobjs,alllosss,allsparse,allcoherent,allflex)
  

  
@tf.function
def jpraw(args,model,x,y,train,bsize):
  allpreds,allzs,allzsum,allzdiff,allpkept,padmask = model(x,train=train,bsize=bsize)
  allcostgs,allcostes,alllosss,allobjs,allcostovers,allsparse,allcoherent = get_loss_sc(args,y,
                                              allpreds,allzs,allzsum,allzdiff,allpkept,padmask)
  vocent = model.get_ratfreq_entropy(x,allzs[0])                                                  
  allflex=[vocent]  
  allcostgs[0]+=args['vocentlam']*(100-vocent)  
  return(allpreds,allzs,allzsum,allzdiff,allpkept,allcostgs,allcostes
        ,alllosss,allobjs,allcostovers,
        allsparse,allcoherent,allflex)

###############################
#######    Encoder  #########
###############################
class Encoder_tran(tf.keras.Model):
    def __init__(self, args,embedding_layer,thetype=None,numhidden=None):      
        # define the layers here
        super().__init__()        
        self.args = args
        if thetype is None:
          thetype = args['etype']
        self.padding_id = tf.convert_to_tensor(
                                              embedding_layer.padding_id,
                                              dtype=tf.int32)
        self.nclasses = nclasses = args['numclass'] 
        self.MAXSEQ=args['max_len'] 
    
        if 'dropout_enc' not in args:
          args['dropout_enc']=args['dropout']    
    
    
        if 'enc_act' not in args or args['enc_act']=='mean_out':
          print('enc mean out')
          self.out_act = self.mean_out
        elif args['enc_act']=='max_out':
          print('enc max out')
          self.out_act = self.max_out
            
        if 'mpe' not in args:
          args['mpe']=args['n_v']    
        if 'tdmodel' in args:
          thehidden = args['tdmodel']
        else:
          thehidden=args['hidden_dimension']
        self.tran = Tran_Encoder(num_layers=args['tnlayers'],
                                             d_model=thehidden,
                                             num_heads=args['theads'],
                                             dff=args['tdff'],
                                             input_vocab_size=args['n_v'],
                                             maximum_position_encoding=args['mpe'],
                                             rate=args['dropout_enc'])  
    def call(self,x,zpred,masks=None,
               training=True,
               dropout=tf.constant(0.0,dtype=tf.float32),
               from_logits=False,temper=1e-5,bsize=None,xyo=None):
        if from_logits:         
          tmask = create_padding_mask(tf.cast(tf.argmax(x,axis=-1),dtype=tf.int32),self.padding_id)          
        else:
          tmask = create_padding_mask(x,self.padding_id)
        masks = tf.expand_dims(masks,2)
        z = tf.expand_dims(zpred,2)
        enc_out = self.tran.call_z(x,training, 
                    tmask,z,from_logits=from_logits,temper=temper)
        rout = self.out_act(enc_out,masks)
        
        return(rout) 
    def max_out(self,enc_out,masks):
      rout = enc_out * masks + (1. - masks) * (-1e6)
      rout = tf.reduce_max(rout, axis=1) 
      return(rout)
    def mean_out(self,enc_out,masks):
      rout = enc_out * masks
      rout = tf.reduce_sum(rout,axis=1)/tf.reduce_sum(masks,axis=1) ## 
      return(rout)
    
        
  
  
########################################################
###############################
#######    Generator  #########
###############################
class Generator2(tf.keras.Model):        
    def __init__(self, args, emblayer):
        super().__init__()
        self.args = args
        self.nclasses = args['numclass']
        self.naspects = len(args['aspects'])        
        n_d = self.args['hidden_dimension'] 
        self.padding_id = tf.convert_to_tensor(
                                              emblayer.padding_id,
                                              dtype=tf.int32)

        # layer list
        self.glayers = []        
        for i in range(2):   
          if 1:
            if 'mpe' not in args:
              args['mpe']=args['n_v']              
            self.glayers.append(Tran_Encoder(num_layers=args['tnlayers'],
                                             d_model=args['hidden_dimension'],
                                             num_heads=args['theads'],
                                             dff=args['tdff'],
                                             input_vocab_size=args['n_v'],
                                             maximum_position_encoding=args['mpe'],####!!!!!!!!
                                             rate=args['dropout'])) 
        self.fcs = [tf.keras.layers.Dense(2,activation=None) for i in range(self.naspects)]
        
       
    #@tf.function    
    def call(self,x,masks,training=True,bsize=None,slevel=None):        
        if slevel is None:
          slevel=self.args['slevel']          
        tmask = create_padding_mask(x,self.padding_id)
        h_concat = self.glayers[0](x,training, 
                    tmask)   
        zs=[];zsums=[];zdiffs=[];pkepts=[];
        for i in range(self.naspects):      
          logits = self.fcs[i](h_concat)
          z,zsum,zdiff,pkept = self.zstuff(logits,masks,training,
                                    bsize=bsize,maxlen=self.args['max_len'],
                                    slevel=slevel)
          zs.append(z);zsums.append(zsum);zdiffs.append(zdiff);pkepts.append(pkept);
        return(zs,zsums,zdiffs,pkepts)
    #@tf.function    
    def call_log(self,x,masks,training=True,bsize=None,slevel=None):        
        if slevel is None:
          slevel=self.args['slevel']  
        tmask = create_padding_mask(tf.cast(tf.argmax(x,axis=-1),dtype=tf.int32),self.padding_id)                    
        h_concat = self.glayers[0](x,training, 
                    tmask,from_logits=True)   
        zs=[];zsums=[];zdiffs=[];pkepts=[];
        for i in range(self.naspects):      
          logits = self.fcs[i](h_concat)
          z,zsum,zdiff,pkept = self.zstuff(logits,masks,training,
                                    bsize=bsize,maxlen=self.args['max_len'],
                                    slevel = slevel)#!!!!!!!
          zs.append(z);zsums.append(zsum);zdiffs.append(zdiff);pkepts.append(pkept);
        return(zs,zsums,zdiffs,pkepts)


    def getL2loss(self,):
      # get l2 cost for all parameters
      lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in self.trainable_variables
                   if 'bias' not in v.name.lower() ]) * self.args['l2_reg']
      return(lossL2)

    @tf.function
    def zstuff(self,z,masks,training=False,bsize=None,maxlen=None,
                   slevel=None):      
      msum = tf.reduce_sum(masks,axis= 1)
      zpass = z[:,:,1]+ (1. - masks) * (-1e6)      
      K = tf.cast(tf.math.round(
              1/tf.reduce_mean(1/(msum*slevel),axis=0)),
            dtype=tf.int32)    
      z_hard = get_top_k_mask(zpass, 
                  K=K,bsize=bsize,maxlen=maxlen)                
      z = tf.stop_gradient(z_hard - z[:,:,1]) + z[:,:,1]    
      zsum = tf.reduce_sum(input_tensor=z,axis=1)/msum      
      zdiff = tf.reduce_mean(tf.reduce_sum(input_tensor=tf.abs(z[:,1:]-z[:,:-1]),axis=1)/msum) 
      pkept = tf.reduce_mean(tf.reduce_sum(z*masks,axis=1)/msum) 
      return(z,zsum,zdiff,pkept)      
def get_top_k_mask(arr,K,bsize=69,maxlen=69):
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
        full_indices, dtype=tf.int64), values=tf.ones_like(values), dense_shape=[bsize,maxlen])
  mask = tf.sparse.to_dense(tf.sparse.reorder(mask_st),default_value=0)  
  return(mask)        
        

######################################################3
class myModel(tf.keras.Model):
  def __init__(self,args,embedding_layer,
               generator=None,encoder=None,outlayer=None):
    super().__init__()
    self.args = args
    self.naspects=len(args['aspects'])
    self.nclasses = nclasses   = args['numclass']
    args,nclasses=self.args,self.nclasses

    if generator is None:
      self.generator = Generator2(args, embedding_layer)
    if 'encoder_hidden' in args:
      numhidden = args['encoder_hidden']
    else:
      numhidden = args['hidden_dimension']
    self.outlayers=[]
    self.encoders=[]
    for a in range(self.naspects):    
      self.outlayers.append(Outlayer(args))
      self.encoders.append(Encoder_tran(args,embedding_layer))
  #@tf.function    
  def call(self,x,y=None,train=True,bsize=None,from_logits=False,
          temper=1e-5,masks=None,slevel=None):
    train=tf.cast(train,dtype=tf.bool)
    if masks is None:
      masks = tf.cast(tf.not_equal(x, self.generator.padding_id),
                        tf.float32,
                        name = 'masks_generator')   
    ## generate highlights
    if from_logits:
      allzs,allzsum,allzdiff,allpkept = self.generator.call_log(x,
                                          masks=masks,training=train,bsize=bsize,
                                          slevel=slevel)
    else:
      allzs,allzsum,allzdiff,allpkept = self.generator(x,
                                          masks=masks,training=train,bsize=bsize,
                                          slevel=slevel)

    allpreds=[]
    for a in range(self.naspects): 
      h_final = self.encoders[a](x,allzs[a],masks,training=train,
                              from_logits=from_logits,temper=temper,bsize=bsize)    
      preds = self.outlayers[a](h_final)
      allpreds.append(preds)              
    return(allpreds,allzs,allzsum,allzdiff,allpkept,masks)



  def one_hotter(self,inds):
    '''
    returns the counts for an integer at that position
    if the integer is bigger than depth, it is not counted
    '''
    out = tf.one_hot(inds,depth=self.args['n_v'])
    out = tf.reduce_sum(tf.reduce_sum(out,axis=0),axis=0)
    return(out)
    
  def split_and_merge(self,vals_mask,padding_id=int(1e10)):
    '''
    this pads with value n_v+1000 which will not be counted in one_hotter
    '''
    out0,out1= tf.dynamic_partition(vals_mask[:,0],tf.cast(vals_mask[:,1],dtype=tf.int32),2)
    return(tf.concat([out1,
                      (out0*0+self.args['n_v']+1000000)],axis=-1)) 

      
    
  def get_ratfreq_entropy(self,x,z):
    '''
    estimate the entropy in the vocab ratfreq
    '''
    xz = tf.concat([tf.expand_dims(tf.cast(x,dtype=tf.int32),axis=-1),
                      tf.expand_dims(tf.cast(z,tf.int32),axis=-1)],
                    axis=-1)  
    inds = tf.map_fn(self.split_and_merge,xz,parallel_iterations=True)
    wcounts = self.one_hotter(inds)
    wfreq = wcounts/tf.reduce_sum(wcounts)
    ent = -tf.reduce_sum(tf.math.multiply_no_nan(tf.math.log(wfreq),wfreq))
    return(ent) 
    
    


def add_default_args(args,rollrandom=True):
  defaultargs = {"log_path":"",
  "load_chkpt_dir":"",
  "train_file":"",
  "dev_file":"",
  "finedev_file":"",
  "fine_file":"",
  "test_file":"",
  "embedding" :"",
  "embdim":100,
  "fixembs":False,
  "hidden_dimension":100,
  
  "tnlayers":4,
  "hidden_dimension":128,
  "tdff":512,
  "theads":8,  
  
  
  
  
  
  "numclass":1,
  "binarize":1,
  "evenup":1,
  "aspects" : [0,1,2,3,4],
  "split_rule":"six",
  "max_len" :256 ,
  "train_batch":64,
  "eval_batch":64,
  "gen_lr":1e-3,
  "enc_lr":1e-3,
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
  "enc_act":"mean_out",
  "trate":5e-4,
  "dropout":0.0,
  "lrdecay":0.99,
  "dosave":1,
  "reload":0,
  "checknum":2000,
  "start_epoch":0,
  "max_epochs":10,
  "min_epochs":5,  
  "beta1":0.9,
  "beta2":0.999,
  "initialization":"rand_uni",
  "edump":0,
  "mtype":"ind_tran",
  "zact":False,
  "n_v":2**15, 
  "logfile" : "logfile.log"}
  newargs = dict(defaultargs)
  theseed = random.randint(1,10000000)
  newargs['rand_seed']=theseed 
  for k in args:
    newargs[k]=args[k]
  if 'dropout_enc' not in newargs:
    newargs['dropout_enc']=newargs['dropout']    
    
  if 'mpe' not in newargs:
    newargs["mpe"]=newargs['n_v']    
  k=0

  try:
    newargs['HOSTNAME']=os.environ['HOSTNAME']
  except:
    newargs['HOSTNAME']=None    

  for i in range(len(newargs['aspects'])):
    for j in range(i+1,len(newargs['aspects'])):
      k+=1
  if k==0:
    k+=1      
  newargs['oversize']=k    
  if newargs['oversize']==0:
    newargs['oversize']=1
  
  return(newargs)

