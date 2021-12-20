import tensorflow as tf
import numpy as np
import random 
import sys
import os
sys.path.append('../share/')
from transformer import create_padding_mask


@tf.function
def ratjdat(args,ratmodel,cfmodel0,cfmodel1,x,y,bsize,
                            opt_rat_enc,opt_rat_gen,opt_cf0,opt_cf1,train=True,
                            nounk=False,unkid=0,doretrace=None):  
    x = tf.cast(x,dtype=tf.int32)
    allpreds,allzs,allzsum,allzdiff,allpkept,padmask = ratmodel(x,train=train,
                                                  bsize=bsize)                                                
    cost_d=0
    dec_ids=0
    newx=0
    newz=allzs[0]
    newy=0
    newpreds=allpreds[0]
    cost_d = get_predloss(allpreds[0],y)

    return(cost_d,dec_ids,
      newx,newz,newy,newpreds)

#######################################################################################################  

@tf.function                
def cag_js2s(args,cfmodel,ratmodel,
               x,y,bsize,opt_cf,train):
  x_nose = remove_start_end(x,cfmodel.start_id,cfmodel.end_id,cfmodel.padding_id)                   
  ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                                             
  _,z,_,_,_,_ = ratmodel(x_nose,train=train,bsize=bsize)## leave on for variety?#!!!!!!!!!!!!!!!!!!!
  ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  if train:
    cost_d,dec_ids=cf_update(args,cfmodel,x,y[:,0],z[0],bsize,opt_cf,train)
  else:

    cost_d,dec_ids=cf_jpredict(args,cfmodel,x,y[:,0],z[0],bsize,train,
                              nounk=False,unkid=0,randsamp=0)
  return(cost_d,dec_ids)

  
def cf_update(args,cfmodel,x,y,z,bsize,opt_cf,train=True):
  if args['hardzero']:    
    x_cfmasked = do_cf_mask(x,y,z,cfmodel.zerotok,cfmodel.zerotok)  
  else:
    x_cfmasked = do_cf_mask(x,y,z,cfmodel.onetok,cfmodel.zerotok)  
  print('xxx', np.shape(x))
  print('zzzz', np.shape(z))
  print('xcfmasked', np.shape(x_cfmasked))
  with tf.GradientTape() as cf_tape:
    logits,dec_ids,masks,cost_d = cfmodel.call_train(x_cfmasked,tar=x,bsize=bsize,
                                  train=train,force=True,
                                  from_logits=False)                                   
    cfvars = cfmodel.trainable_variables
    gradientscf = cf_tape.gradient(cost_d,cfvars)
    opt_cf.apply_gradients(zip(gradientscf,cfvars))
  return(cost_d,dec_ids)  

###################################################################################################
@tf.function
def jpredict_by1(args,ratmodel,cfmodel,x,y,bsize,train=True,
                            nounk=False,unkid=0,randsamp=0,
                            doretrace=None):  
  x_nose = remove_start_end(x,cfmodel.start_id,cfmodel.end_id,cfmodel.padding_id)                                                                
  allpreds,z,_,_,_,_ = ratmodel(x_nose,train=train,bsize=bsize,slevel=args['slevel'])      
  z = z[0]
  y=y[:,0]
  if args['iterdecode']:
    masks = tf.cast(tf.not_equal(x_nose, cfmodel.padding_id),
                      tf.float32,
                      name = 'masks_generator')  
    msum = tf.reduce_sum(masks,axis= 1)
    K = tf.cast(tf.math.round(1/tf.reduce_mean(1/(msum*args['slevel']),axis=0)),
            dtype=tf.int32)      
    cost_d,dec_ids = cf_jpredict_dynamic(args,cfmodel,
                                    x,y,z,tf.shape(x)[0],K,train,
                                  nounk=nounk,unkid=unkid,randsamp=randsamp)
    
  else:  
    cost_d,dec_ids = cf_jpredict(args,cfmodel,x,y,z,tf.shape(x)[0],train,
                                  nounk=nounk,unkid=unkid,randsamp=randsamp)    
    

  ## YO
  dec_ids = remove_start_end(tf.cast(dec_ids,dtype=tf.int32),
                        cfmodel.start_id,cfmodel.end_id,cfmodel.padding_id)                                                                                                                                  
  ## YO
  newx = x_nose
  newz = z
  newy = y
  newpreds = allpreds[0]

  return(cost_d,dec_ids,
      newx,newz,newy,newpreds)   



###################################################################################################
@tf.function
def jpredict_js2s(args,ratmodel,cfmodel0,cfmodel1,x,y,bsize,
                            opt_rat_enc,opt_rat_gen,opt_cf0,opt_cf1,train=True,
                            nounk=False,unkid=0,randsamp=0,
                            doretrace=None):  
  '''                            
  
  '''          
  moo=doretrace       
  x_nose = remove_start_end(x,cfmodel0.start_id,cfmodel0.end_id,cfmodel0.padding_id)                                                                
  
  allpreds,z,_,_,_,_ = ratmodel(x_nose,train=train,bsize=bsize)    
    
  
  x0,x1 = tf.dynamic_partition(data=x,partitions=tf.cast(y[:,0],dtype=tf.int32),
                                  num_partitions=2)  
  z0,z1 = tf.dynamic_partition(data=z[0],partitions=tf.cast(y[:,0],dtype=tf.int32),
                                  num_partitions=2)  
  y0 = tf.zeros(shape=tf.shape(x0)[0],dtype=tf.float32)
  y1 = tf.ones(shape=tf.shape(x1)[0],dtype=tf.float32)  
  preds0,preds1 = tf.dynamic_partition(data=allpreds[0],
                      partitions=tf.cast(y[:,0],dtype=tf.int32),num_partitions=2)    
  
  print('x0', np.shape(x0), 'x1', np.shape(x1))
  print('y0', np.shape(y0),'y1', np.shape(y1))
  if args['iterdecode']:
    masks = tf.cast(tf.not_equal(x_nose, cfmodel0.padding_id),
                      tf.float32,
                      name = 'masks_generator')  
    msum = tf.reduce_sum(masks,axis= 1)
    K = tf.cast(tf.math.round(1/tf.reduce_mean(1/(msum*args['slevel']),axis=0)),
            dtype=tf.int32)
      
    zsum =tf.reduce_sum(z0,axis=-1)
    
    cost_d0,dec_ids0 = cf_jpredict_dynamic(args,cfmodel0,x0,y0,z0,tf.shape(x0)[0],K,train,
                                  nounk=nounk,unkid=unkid,randsamp=randsamp)
    zsum =tf.reduce_sum(z1,axis=-1)
    
    cost_d1,dec_ids1 = cf_jpredict_dynamic(args,cfmodel1,x1,y1,z1,tf.shape(x1)[0],K,train,
                                  nounk=nounk,unkid=unkid,randsamp=randsamp)    
  else:  
    cost_d0,dec_ids0 = cf_jpredict(args,cfmodel0,x0,y0,z0,tf.shape(x0)[0],train,
                                  nounk=nounk,unkid=unkid,randsamp=randsamp)
    cost_d1,dec_ids1 = cf_jpredict(args,cfmodel1,x1,y1,z1,tf.shape(x1)[0],train,
                                    nounk=nounk,unkid=unkid,randsamp=randsamp)    
  num0 = tf.cast(tf.shape(x0)[0],dtype=tf.float32)
  num1 = tf.cast(tf.shape(x1)[0],dtype=tf.float32)
  cost_d0 = tf.cast(cost_d0,dtype=tf.float32)
  cost_d1 = tf.cast(cost_d1,dtype=tf.float32)
   
  print('costd0', np.shape(cost_d0),'cost_d1', np.shape(cost_d1), cost_d0.dtype, cost_d1.dtype)                                                        
  

  if num0==0:
    cost_d0=tf.cast(0,dtype=tf.float32)
  if num1==0:
    cost_d1=tf.cast(0,dtype=tf.float32)
  cost_d = (cost_d0*num0 + cost_d1*num1)/(num0+num1)
  
  dec_ids = tf.concat([dec_ids0,dec_ids1],axis=0)


  newx = tf.concat([x0,x1],axis=0)
  newz = tf.concat([z0,z1],axis=0)  
  newy = tf.concat([y0,y1],axis=0)  
  newpreds = tf.concat([preds0,preds1],axis=0)  



  ## YO
  newx = remove_start_end(newx,
                        cfmodel0.start_id,cfmodel0.end_id,cfmodel0.padding_id)
  dec_ids = remove_start_end(tf.cast(dec_ids,dtype=tf.int32),
                        cfmodel0.start_id,cfmodel0.end_id,cfmodel0.padding_id)                                                                                                                                  
  ## YO

  return(cost_d,dec_ids,
      newx,newz,newy,newpreds)   

##################################       
def cf_jpredict(args,cfmodel,x,y,z,bsize,train=False,nounk=False,unkid=0,randsamp=0):  

  #### replace the rationale with the mask tokens
  if args['hardzero']:    
    x_cfmasked = do_cf_mask(x,y,z,cfmodel.zerotok,cfmodel.zerotok)  
  else:
    x_cfmasked = do_cf_mask(x,y,z,cfmodel.onetok,cfmodel.zerotok)  
  if nounk:
    logits,dec_ids,masks,cost_d = cfmodel.call_eval_nounk(x_cfmasked,tar=x,bsize=bsize,
                                  train=train,force=True,
                                  from_logits=False) 
  else:  
    logits,dec_ids,masks,cost_d = cfmodel.call_eval(x_cfmasked,tar=x,bsize=bsize,
                                  train=train,force=True,
                                  from_logits=False)  
  if randsamp:
    tf.print('rand samp homie!!')
    dec_ids = tf.map_fn(fn=lambda t: tf.random.categorical(logits=t,num_samples=1),
                      elems=tf.transpose(logits,[1,0,2]), ## map goes on axis 0
                      dtype=tf.int64,parallel_iterations=True) 
    dec_ids = tf.transpose(dec_ids,[1,0,2])[:,:,0]   ##go back to B,T
  return(cost_d,dec_ids) 






#@tf.function 
def cf_jpredict_dynamic(args,cfmodel,x,y,z,bsize,K,
            train=False,nounk=False,unkid=0,randsamp=0):  
  print('CF JPREDICT DYNAMIC')
  zinds = tf.map_fn(lambda x:tf.cast(tf.where(x==1),dtype=tf.int32),
               tf.cast(tf.math.round(z),dtype=tf.int32),
               parallel_iterations=True)
  zinds = zinds+1 ## z is shifted from this x by startid!!!!!!               
  zinds = zinds[:,:,0]
  if args['hardzero']:    
    x_cfmasked = do_cf_mask(x,y,z,cfmodel.zerotok,cfmodel.zerotok)  
  else:
    x_cfmasked = do_cf_mask(x,y,z,cfmodel.onetok,cfmodel.zerotok)  
  
  cost_d=0.0
  for k in range(K):    
    ## get new dec_ids
    logits,dec_ids,masks,cost_d = cfmodel.call_eval(x_cfmasked,tar=x,bsize=bsize,
                                  train=train,force=True,
                                  from_logits=False)      
    if randsamp:
      print('rand samp homie!!')
      dec_ids = tf.map_fn(fn=lambda t: tf.random.categorical(logits=t,num_samples=1),
                        elems=tf.transpose(logits,[1,0,2]), ## map goes on axis 0
                        dtype=tf.int64,parallel_iterations=True) 
      dec_ids = tf.transpose(dec_ids,[1,0,2])[:,:,0]   ##go back to B,T

    ## add it back in    
    keepmask = tf.cast(tf.squeeze(tf.one_hot(zinds[:,k],
                        depth=tf.shape(x)[1])),dtype=tf.int32)    
    x_cfmasked = x_cfmasked*(1-keepmask) + tf.cast(dec_ids,dtype=tf.int32)*(keepmask)                                
    x_cfmasked = tf.reshape(x_cfmasked,shape=tf.shape(x))
  return(cost_d,x_cfmasked) 


def jpredict_wdat(args,ratmodel,cfmodel,x,y,train=False,nounk=False,unkid=0,bsize=0):
  x_nose = remove_start_end(x,cfmodel.start_id,cfmodel.end_id,cfmodel.padding_id)                                                                
  allpreds,z,_,_,_,_ = ratmodel(x_nose,train=train,bsize=bsize)    
  z = z[0]
  if args['toclass']=='positive':
      the_y = tf.ones_like(y[:,0])
  else:
      the_y = tf.zeros_like(y[:,0])

  if args['hardzero']:    
    x_cfmasked = do_cf_mask(x,the_y,z,cfmodel.zerotok,cfmodel.zerotok)  
  else:
    x_cfmasked = do_cf_mask(x,the_y,z,cfmodel.onetok,cfmodel.zerotok)  
  if nounk:
    logits,dec_ids,masks,cost_d = cfmodel.call_eval_nounk(x_cfmasked,tar=x,bsize=bsize,
                                  train=train,force=True,
                                  from_logits=False) ## should pass unkid  
  else:  
    logits,dec_ids,masks,cost_d = cfmodel.call_eval(x_cfmasked,tar=x,bsize=bsize,
                                  train=train,force=True,
                                 from_logits=False)  
  dec_ids = tf.cast(dec_ids,tf.int32) 
  x_nose_cf = remove_start_end(dec_ids,cfmodel.start_id,cfmodel.end_id,cfmodel.padding_id)                                                                  
  x_nose_cf = tf.cast(x_nose_cf,tf.float32)
  z = tf.cast(z,tf.float32)
  x_nose = tf.cast(x_nose,tf.float32)
  x_nose_cf = x_nose_cf*z + x_nose*(1-z)  
  x_nose_cf = tf.cast(x_nose_cf,tf.int32)
  
  allpreds_cf,z_cf,_,_,_,_ = ratmodel(x_nose_cf,train=train,bsize=bsize)    
  z_cf=z_cf[0]
  return(cost_d,
         x_nose,allpreds,z,
         x_nose_cf,allpreds_cf,z_cf)   
      




@tf.function                
def disceronly_GAN(args,cfmodel,discer,ratmodel,
               x,y,bsize,opt_cf,opt_discer,train,ganlambda):
  x_nose = remove_start_end(x,cfmodel.start_id,cfmodel.end_id,cfmodel.padding_id)                                                                
  allpreds,z,_,_,_,_ = ratmodel(x_nose,train=train,bsize=bsize)    
  if args['toclass']=='positive':
      the_y = tf.zeros_like(y)
  else:
      the_y = tf.ones_like(y)
  cfloss,cfpredloss,ganloss=JD_update(args,ganlambda,cfmodel,discer,ratmodel,
                                            x,the_y,z[0],
                                            bsize,opt_cf,opt_discer,train) 
  return(cfloss,cfpredloss,ganloss)   

  
#######################################################################################################

def disceronly_update(args,ganlambda,cfmodel,discer,ratmodel,x,y,z,bsize,opt_cf,opt_disc,
      train=True,from_logits=True,temper=1e-5):   
  ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  y = 1-y #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ###!!!!!!!!!!!!!!!!!!!!!!!!!!!    
  print('disceronly', np.shape(x), np.shape(y))  
  x_cfmasked = do_cf_mask(x,y[:,0],z,cfmodel.zerotok,cfmodel.zerotok)  
  
  x_nose = remove_start_end(x,cfmodel.start_id,cfmodel.end_id,cfmodel.padding_id)                                                                  
  padmask = 1-tf.cast(tf.math.equal(x_nose, cfmodel.padding_id),dtype=tf.float32)  
  with  tf.GradientTape() as disc_tape:
    logits = cfmodel(x_cfmasked,tar=x,bsize=bsize,
                                  train=train) 
    logits = tf.nn.softmax(logits,axis=-1)
    newlogits = remove_start_end_logits(x,logits,cfmodel.start_id, ## careful with start end toks
                                                      cfmodel.end_id,
                                                      cfmodel.padding_id) 
    ## grab og data when not z, wasnt on for ganbigbeer0                                                   
    newlogits= grab_og_data(newlogits,x_nose,z,args['n_v']) ## keep og data when not z                                                      
    ###################################################3
    cfpredloss = 0       
    ###################################################3
    ## get GAN loss
    x_hot = tf.one_hot(x_nose,depth=args['n_v'])
    if args['padnotrat']:
      print('PADNOTRAT')
      padmask2 = tf.concat([z,z],axis=0)              
    else:
      padmask2 = tf.concat([padmask,padmask],axis=0)  
      thez=0
      
    
    ganx = tf.cast(tf.concat([x_hot,newlogits],axis=0),dtype=tf.float32)
    gany = tf.cast(tf.concat([tf.ones_like(y),tf.zeros_like(y)],axis=0),
              dtype=tf.float32)
    discpred = discer(ganx,padmask2,train,from_logits,temper)
    ganloss = get_predloss(discpred,gany[:,0]) 
    cfloss = 0
    discvars = discer.trainable_variables
    gradientsdisc = disc_tape.gradient(ganloss,discvars)
    opt_disc.apply_gradients(zip(gradientsdisc,discvars))
  return(cfloss,cfpredloss,ganloss)



@tf.function                
def single_GANbyc(args,cfmodel,discer,ratmodel,
               x,y,bsize,opt_cf,opt_discer,train,ganlambda):
  x_nose = remove_start_end(x,cfmodel.start_id,cfmodel.end_id,cfmodel.padding_id)                                                                
  allpreds,z,_,_,_,_ = ratmodel(x_nose,train=train,bsize=bsize)    

  
  if args['toclass']=='positive':
      the_y = tf.zeros_like(y)
  else:
      the_y = tf.ones_like(y)
  
  if train:

    cfloss,cfpredloss,ganloss=GANbyc_update(args,ganlambda,cfmodel,discer,ratmodel,
                                            x,the_y,z[0],y,
                                            bsize,opt_cf,opt_discer,train) 
    
  else:
    cfloss,cfpredloss,ganloss=GAN_jpredict(args,cfmodel,discer,ratmodel,
                                            x,the_y,z[0],y,
                                            bsize,opt_cf,opt_discer,train) 
    
  return(cfloss,cfpredloss,ganloss)    
#@tf.function
def GANbyc_update(args,ganlambda,cfmodel,discer,ratmodel,x,y,z,truey,bsize,opt_cf,opt_disc,
      train=True,from_logits=True,temper=1e-5):   
  '''
  we pit the flipped samples against the ground truth samples. this way we cant just 
  predict positive from true....
  this will have an uneven amount of 1s and 0s in each batch, thats ok. it evens out
  ''' 
  ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  y = 1-y #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ###!!!!!!!!!!!!!!!!!!!!!!!!!!!    
  x_cfmasked = do_cf_mask(x,y[:,0],z,cfmodel.zerotok,cfmodel.zerotok)  

  x_nose = remove_start_end(x,cfmodel.start_id,cfmodel.end_id,cfmodel.padding_id)                                                                  
  padmask = 1-tf.cast(tf.math.equal(x_nose, cfmodel.padding_id),dtype=tf.float32)  
  with tf.GradientTape() as cf_tape, tf.GradientTape() as disc_tape:
    logits = cfmodel(x_cfmasked,tar=x,bsize=bsize,
                                  train=train) 
    logits = tf.nn.softmax(logits,axis=-1)
    newlogits = remove_start_end_logits(x,logits,cfmodel.start_id, ## careful with start end toks
                                                      cfmodel.end_id,
                                                      cfmodel.padding_id) 
    newlogits= grab_og_data(newlogits,x_nose,z,args['n_v']) ## keep og data when not z                                                      
    ###################################################3
    ## get RL loss
    allpreds,_,_,_,_,_=ratmodel(newlogits,train=args['GRtrain'],bsize=bsize, 
                                  from_logits=from_logits,temper=temper,masks=padmask)

    cfpredloss = get_predloss(allpreds[0],y[:,0])  
    ## get GAN loss
    x_hot = tf.one_hot(x_nose,depth=args['n_v'])
    if args['padnotrat']:
      print('PADNOTRAT')
      
      x_hot = pad_not_rat(x_nose,x_hot,z,cfmodel.padding_id)
      newlogits = pad_not_rat(x_nose,newlogits,z,cfmodel.padding_id)
      padmask2 = tf.concat([z,z],axis=0)              
    else:
      padmask2 = tf.concat([padmask,padmask],axis=0)  
      

      
    
    ganx = tf.cast(tf.concat([x_hot,newlogits],axis=0),dtype=tf.float32)
    gany = tf.cast(tf.concat([tf.ones_like(y),tf.zeros_like(y)],axis=0),
              dtype=tf.float32)
    discpred = discer(ganx,padmask2,train,from_logits,temper)
    
    if args['toclass']=='positive':
      gmask = tf.concat([truey[:,0],1-truey[:,0]],axis=0)
    else:
      gmask = tf.concat([1-truey[:,0],truey[:,0]],axis=0)
    
    ganloss = get_predloss_wmask(discpred,gany[:,0],gmask)
                        
    cfvars = cfmodel.trainable_variables    
    cfloss = args['RLlambda']*cfpredloss - ganlambda*ganloss
    gradientscf = cf_tape.gradient(cfloss,cfvars)

    discvars = discer.trainable_variables
    gradientsdisc = disc_tape.gradient(ganloss/ganlambda,discvars) 

    ## apply the gradients
    opt_cf.apply_gradients(zip(gradientscf,cfvars))
    opt_disc.apply_gradients(zip(gradientsdisc,discvars))
  return(cfloss,cfpredloss,ganloss)



def JD_update(args,ganlambda,cfmodel,discer,ratmodel,x,y,z,bsize,opt_cf,opt_disc,
      train=True,from_logits=True,temper=1e-5):   
  ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  y = 1-y #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ###!!!!!!!!!!!!!!!!!!!!!!!!!!!  
  
  x_cfmasked = do_cf_mask(x,y[:,0],z,cfmodel.zerotok,cfmodel.zerotok)  
  

  x_nose = remove_start_end(x,cfmodel.start_id,cfmodel.end_id,cfmodel.padding_id)                                                                  
  padmask = 1-tf.cast(tf.math.equal(x_nose, cfmodel.padding_id),dtype=tf.float32)  
  with  tf.GradientTape() as disc_tape:
    logits = cfmodel(x_cfmasked,tar=x,bsize=bsize,
                                  train=train) 
    logits = tf.nn.softmax(logits,axis=-1)
    newlogits = remove_start_end_logits(x,logits,cfmodel.start_id, ## careful with start end toks
                                                      cfmodel.end_id,
                                                      cfmodel.padding_id) 
    newlogits= grab_og_data(newlogits,x_nose,z,args['n_v']) ## keep og data when not z                                                      
    ###################################################3
    cfpredloss = 1.25
    ###################################################3
    ## get GAN loss
    x_hot = tf.one_hot(x_nose,depth=args['n_v'])
    if args['padnotrat']:
      print('PADNOTRAT')
      
      x_hot = pad_not_rat(x_nose,x_hot,z,cfmodel.padding_id)
      newlogits = pad_not_rat(x_nose,newlogits,z,cfmodel.padding_id)
      padmask2 = tf.concat([z,z],axis=0)              
    else:
      padmask2 = tf.concat([padmask,padmask],axis=0)  
      
    
    ganx = tf.cast(tf.concat([x_hot,newlogits],axis=0),dtype=tf.float32)
    gany = tf.cast(tf.concat([tf.ones_like(y),tf.zeros_like(y)],axis=0),
              dtype=tf.float32)
    discpred = discer(ganx,padmask2,train,from_logits,temper)    
    ganloss = get_predloss(discpred,gany[:,0])
    ##########################################3
    discvars = discer.trainable_variables
    
    gradientsdisc = disc_tape.gradient(ganloss/ganlambda,discvars) 
    opt_disc.apply_gradients(zip(gradientsdisc,discvars))
    cfloss=1.69
  return(cfloss,cfpredloss,ganloss)
#######################################################################################################
def pad_not_rat(x,logits,z,padid):
  '''
  replace everything thats not rat with padid
  '''
  
  z=tf.cast(tf.expand_dims(z,axis=-1),dtype=logits.dtype)
  padhot=tf.one_hot(tf.ones_like(x,dtype=tf.int32)*tf.cast(padid,dtype=tf.int32),
                        depth=tf.shape(logits)[-1],dtype=tf.float32)
  print('logits', np.shape(logits), 'z', np.shape(z), 'padhot', np.shape(padhot))
  
  out = logits*z+(1-z)*padhot
  return(out)
  
  
def do_cf_mask(x,y,z,tok0,tok1):
  '''
  x2 = [keep everything where z is 0 and everything else to zero]
      + [the ind of interest at the z1 spots]
  but you only do this to y==[1/0] at a time    
  
  
  what you pass in y=1-ytrue
  then at this current y, you replace the y=0 with tok0
  so tok0 means I WANT THE OUTPUT TO BE y=0

  x has start end
  x_nos goes to ratmodel.... we get z
  x_cf has start end
  
  z saw a version of x that was shifted and end was pad
  the last token in x_nos is always pad
  here we have to shift everything back to the right
  put a zero in the front, and drop the last z which is always zero
  '''

  ## line z up to x
  zerov = tf.zeros(shape=(tf.shape(x)[0],1),dtype=z.dtype)
  z = tf.concat([zerov,z[:,:-1]],axis=-1)
  y=tf.cast(y,dtype=tf.int32)
  z=tf.cast(z,dtype=tf.int32)
  x2 = tf.expand_dims(tf.cast(tf.math.equal(y,0),dtype=tf.int32),axis=-1)*(x*(1-z) + tok0*z) + \
       tf.expand_dims(tf.cast(tf.math.not_equal(y,0),dtype=tf.int32),axis=-1)*x


  x3 = tf.expand_dims(tf.cast(tf.math.equal(y,1),dtype=tf.int32),axis=-1)*(x2*(1-z) + tok1*z) + \
       tf.expand_dims(tf.cast(tf.math.not_equal(y,1),dtype=tf.int32),axis=-1)*x2
  return(x3)

def remove_start_end(x,startid,endid,padid):
  '''
  we first remove first token which should be <start>
  we concat on a <pad> to the end
  we replace <end> with pad
  '''

  ## remove first token, assuming its <start> and 
  ## add a <pad> to the end
  x = x[:,1:] ## remove first token
  padv = tf.zeros(shape=(tf.shape(x)[0],1),dtype=x.dtype)+padid
  x = tf.concat([x,padv],axis=-1)
  ## replace the <end> token with <pad>
  mask_end = tf.cast(tf.equal(x, endid),
                      tf.int32,
                      name = 'masks_generator')
  x3 = (x*(1-mask_end) + padid*mask_end)

  
  return(x3)


def remove_start_end_logits(x,logits,startid,endid,padid):
  '''
  this first replaces <end> with <pad>
  we then remove the first token which should be <start>
  we then concat on a <pad> token
  '''
  mask_end = tf.cast(tf.equal(x, endid),
                      tf.float32,
                      name = 'masks_generator')
  mask_end=tf.expand_dims(mask_end,axis=-1)                      
  
                 
  padhot=tf.one_hot(tf.ones_like(x)*padid,depth=tf.shape(logits)[-1],
                                                  dtype=tf.float32)##!!!
  logits = (logits*(1-mask_end) + padhot*mask_end)  
  

  ## remove first token, assuming its <start> and 
  ## add a <pad> to the end
  logits = logits[:,1:] ## remove first token
  padv  = tf.expand_dims(padhot[:,0,],axis=1)
  logits = tf.concat([logits,padv],axis=1)

  return(logits)
  
 

def get_predloss(preds,y):
  
  loss_mat = tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=preds,
                                                      from_logits=0)
  
  loss =  tf.reduce_mean(input_tensor=loss_mat,axis=0)
  
  return(loss)  
def get_predloss_wmask(preds,y,amask):
  
  loss_mat = tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=preds,
                                                      from_logits=0)
  
  loss = tf.reduce_sum(input_tensor=loss_mat*amask,axis=0)
  loss = loss/tf.reduce_sum(amask,axis=0)
  
  return(loss)   

def grab_og_data(dlogits,x,z,n_v):
  xhot = tf.one_hot(x,depth=n_v,axis=-1,dtype=tf.float32) 
  z=tf.expand_dims(z,axis=-1)
  newlogits = z*dlogits + (1-z)*xhot
  return(newlogits)

#######################################################################################################
def add_default_args(args,rollrandom=True):
  defaultargs = {"log_path":"",
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
  'GRtrain':False,
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
  "epochs_discer":0,
  "epochs_jcf":30,
  "max_epochs":30,
  "min_epochs":5,
  "beta1":0.9,
  "beta2":0.999,
  "initialization":"rand_uni",
  "edump":0,
  "mtype":"fcbert",
  "masktype":"cont",
  "classpick":-1,
  "hardzero":0,
  "padnotrat":0,
  "zact":False,
  "cf_lambda":1e-3, 
  "chk_obj_mult":1.0,
  "GANlambda":1,
  "RLlambda":1,  
  "Zlambda":1,
  "n_v":2**15, 
  "rlsup":0,
  'iterdecode':0,
  "logfile" : "logfile.log"}
  newargs = dict(defaultargs)
  theseed = random.randint(1,10000000)
  newargs['rand_seed']=theseed ## this will be overwritten if in args
  try:
    newargs['HOSTNAME']=os.environ['HOSTNAME']
  except:
    newargs['HOSTNAME']=None 
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
  
  if newargs['toclass']=='positive':
    newargs['toint']=1
  else:
    newargs['toint']=0

  return(newargs)      
        

class Discer(tf.keras.Model):
  def __init__(self,args,encoder):
    super().__init__()
    self.encoder=encoder
    self.outlayer = tf.keras.layers.Dense(2,activation='softmax')
  def call(self,x,padmask=None,train=True,from_logits=False,temper=1e-5,bsize=None):
    if from_logits:
      z = tf.cast(tf.ones_like(x[:,:,0]),dtype=x.dtype)
    else:
      z = tf.cast(tf.ones_like(x),dtype=tf.float32)
    enc_out = self.encoder(x,z,masks=padmask,training=train,
                  from_logits=from_logits,temper=temper,bsize=bsize)  
    xout = self.outlayer(enc_out)
    return(xout)


