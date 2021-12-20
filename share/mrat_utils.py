import json
import sys
import tensorflow as tf
from ratstuff import *



def load_jrat(args,embed_layer,myModel,chkptdir=None):
  amodel = myModel(args,embed_layer)
   
  theckpt = tf.train.Checkpoint(step=tf.Variable(1),
                              net=amodel)
  chkptman = tf.train.CheckpointManager(theckpt,
                                        args['log_path'],
                                        max_to_keep=1)
  if len(args['load_chkpt_dir'])>0 and chkptdir is None:
    chkptdir=args['load_chkpt_dir']
  if chkptdir is not None:
    theckpt.restore(
                tf.train.latest_checkpoint(chkptdir)
                            ).assert_nontrivial_match()
    print('LOADED MODEL', chkptdir)
  if args['gen_lr'] == -1:
    glr =  CustomSchedule(args['hidden_dimension'])
  else:  
    glr = tf.Variable(args['gen_lr'],dtype=tf.float32)
  if args['enc_lr'] ==  -1:
    elr =  CustomSchedule(args['hidden_dimension'])  
  else:
    elr = tf.Variable(args['enc_lr'],dtype=tf.float32)

  ogen0 = tf.keras.optimizers.Adam(learning_rate=glr,
                                  beta_1 = args['beta1'],
                                  beta_2 = args['beta2']) 
  oenc0 = tf.keras.optimizers.Adam(learning_rate=elr)
  
  
  ogen1 = tf.keras.optimizers.Adam(learning_rate=glr,
                                  beta_1 = args['beta1'],
                                  beta_2 = args['beta2']) 
  oenc1 = tf.keras.optimizers.Adam(learning_rate=elr)


  ogen2 = tf.keras.optimizers.Adam(learning_rate=glr,
                                  beta_1 = args['beta1'],
                                  beta_2 = args['beta2']) 
  oenc2 = tf.keras.optimizers.Adam(learning_rate=elr)

  
  return(amodel,theckpt,chkptman,ogen0,oenc0,ogen1,oenc1,ogen2,oenc2)


def get_loss(args,y,allpreds,allzs,allzsum,allzdiff,allpkept,padmask):
  allgs=[];alles=[];alllosss=[];allobjs=[];
  ## generator and encoder loss
  for i in range(len(args['aspects'])):    
    cost_g,cost_e,loss,obj = compute_loss_gen_enc(args=args,
                  preds=allpreds[i],
                  y=y[:,i],
                  zsum=allzsum[i],
                  zdiff=allzdiff[i],
                  )     
    allgs.append(cost_g);alles.append(cost_e);alllosss.append(loss);allobjs.append(obj);
    

  pmsum = tf.reduce_sum(padmask,axis=1)
  allovers = tf.TensorArray(tf.float32,size=args['oversize']) ###########!!!!!!!!!!!!!
  if len(args['aspects'])>1:
    k=0
    for i in range(len(args['aspects'])):
      for j in range(i+1,len(args['aspects'])):  
        oi = tf.reduce_mean(tf.reduce_sum(allzs[i]*allzs[j],axis=1)/pmsum)      ## o
        allovers=allovers.write(k,oi)
        k+=1      
  else:
      allovers=allovers.write(0,0.0)
  
  allovers=allovers.stack()
  return(allgs,alles,alllosss,allobjs,allovers)
  
def get_loss_sc(args,y,allpreds,allzs,allzsum,allzdiff,allpkept,padmask):
  allgs=[];alles=[];alllosss=[];allobjs=[];allsparse=[];allcoherent=[];
  ## generator and encoder loss
  for i in range(len(args['aspects'])):    
    cost_g,cost_e,loss,obj,sparsity,coherent = compute_loss_single(args=args,
                  preds=allpreds[i],
                  y=y[:,i],
                  zsum=allzsum[i],
                  zdiff=allzdiff[i],
                  )     
    allgs.append(cost_g);alles.append(cost_e);alllosss.append(loss);allobjs.append(obj);
    allsparse.append(sparsity);allcoherent.append(coherent);

  pmsum = tf.reduce_sum(padmask,axis=1)
  allovers = tf.TensorArray(tf.float32,size=args['oversize']) 
  if len(args['aspects'])>1:
    k=0
    for i in range(len(args['aspects'])):
      for j in range(i+1,len(args['aspects'])):  
        oi = tf.reduce_mean(tf.reduce_sum(allzs[i]*allzs[j],axis=1)/pmsum)      ## og      
        allovers=allovers.write(k,oi)
        k+=1        
  else:
      allovers=allovers.write(0,0.0)
  allovers=allovers.stack()
  return(allgs,alles,alllosss,allobjs,allovers,allsparse,allcoherent)
  
    

def cag_wrap(compute_apply_gradients,args,model,x,y,
              optimizer_gen0,optimizer_enc0,
              optimizer_gen1,optimizer_enc1,
              optimizer_gen2,optimizer_enc2,                            
                            train):
  (allcostgs,allcostes,allzsum,
   allzdiff,allpkept,allcostovers,allobjs,alllosss,
       allsparse,allcoherent,allflex) = compute_apply_gradients(args,model,x,y,
                                                          optimizer_gen0,optimizer_enc0,
                                                          optimizer_gen1,optimizer_enc1,
                                                          optimizer_gen2,optimizer_enc2,
                                                          train)
  ddict={}
  for i in range(len(args['aspects'])):
    ddict['cost_g'+str(i)]=allcostgs[i].numpy()                                                          
    ddict['cost_e'+str(i)]=allcostes[i].numpy()    
    ddict['loss'+str(i)]=alllosss[i].numpy()
    ddict['obj'+str(i)]=allobjs[i].numpy()    
    ddict['pkept'+str(i)]=allpkept[i].numpy()    
    ddict['sparse'+str(i)]=allsparse[i].numpy()        
    ddict['coherency'+str(i)]=allcoherent[i].numpy()  
    ddict['flex'+str(i)]=allflex[i].numpy()              
  for j in range(len(allcostovers)):
    ddict['over'+str(j)]=allcostovers[j].numpy()

  return(ddict)

def cag_wrap_fix(compute_apply_gradients,args,model,x,y,
              optimizer_gen0,optimizer_enc0,                                         
                            train,bsize):
  (allcostgs,allcostes,allzsum,
   allzdiff,allpkept,allcostovers,allobjs,alllosss,
  allsparse,allcoherent,allflex) = compute_apply_gradients(args,model,x,y,
                                                optimizer_gen0,optimizer_enc0,
                                                train,bsize=bsize)
  ddict={}
  for i in range(len(args['aspects'])):
    ddict['cost_g'+str(i)]=allcostgs[i].numpy()                                                          
    ddict['cost_e'+str(i)]=allcostes[i].numpy()    
    ddict['loss'+str(i)]=alllosss[i].numpy()
    ddict['obj'+str(i)]=allobjs[i].numpy()    
    ddict['pkept'+str(i)]=allpkept[i].numpy()    
    ddict['sparse'+str(i)]=allsparse[i].numpy()        
    ddict['coherency'+str(i)]=allcoherent[i].numpy()  
    ddict['flex'+str(i)]=allflex[i].numpy()  
    
  for j in range(len(allcostovers)):
    ddict['over'+str(j)]=allcostovers[j].numpy()

  return(ddict)  
  
  
def just_predict(jpraw,args,model,x,y,train): 
  
  (allpreds,allzs,allzsum,allzdiff,allpkept,
   allcostgs,allcostes,alllosss,allobjs,allcostovers,
       allsparse,allcoherent,allflex) = jpraw(args,model,x,y,train)
  ddict={}
  for i in range(len(args['aspects'])):
    ddict['cost_g'+str(i)]=allcostgs[i].numpy()                                                          
    ddict['cost_e'+str(i)]=allcostes[i].numpy()    
    ddict['loss'+str(i)]=alllosss[i].numpy()
    ddict['obj'+str(i)]=allobjs[i].numpy()    
    ddict['pkept'+str(i)]=allpkept[i].numpy()    
    ddict['sparse'+str(i)]=allsparse[i].numpy()        
    ddict['coherency'+str(i)]=allcoherent[i].numpy()  
    ddict['flex'+str(i)]=allflex[i].numpy()                  
  for j in range(len(allcostovers)):
    ddict['over'+str(j)]=allcostovers[j].numpy()
  return(ddict)
  
def just_predict_fix(jpraw,args,model,x,y,train,bsize): 

  (allpreds,allzs,allzsum,allzdiff,allpkept,
   allcostgs,allcostes,alllosss,allobjs,allcostovers,
       allsparse,allcoherent,allflex) = jpraw(args,model,x,y,train,bsize=bsize)
  ddict={}
  for i in range(len(args['aspects'])):
    ddict['cost_g'+str(i)]=allcostgs[i].numpy()                                                          
    ddict['cost_e'+str(i)]=allcostes[i].numpy()    
    ddict['loss'+str(i)]=alllosss[i].numpy()
    ddict['obj'+str(i)]=allobjs[i].numpy()    
    ddict['pkept'+str(i)]=allpkept[i].numpy()    
    ddict['sparse'+str(i)]=allsparse[i].numpy()        
    ddict['coherency'+str(i)]=allcoherent[i].numpy()  
    ddict['flex'+str(i)]=allflex[i].numpy()  
    
  for j in range(len(allcostovers)):
    ddict['over'+str(j)]=allcostovers[j].numpy()

  return(ddict)

  
def just_predict_wdat(jpraw,args,model,x,y,train,bsize=None):   
  (allpreds,allzs,allzsum,allzdiff,allpkept,
    allcostgs,allcostes,alllosss,allobjs,allcostovers,
    allsparse,allcoherent,allflex) = jpraw(args,model,x,y,train,bsize=bsize)
  ddict={}
  for i in range(len(args['aspects'])):
    ddict['cost_g'+str(i)]=allcostgs[i].numpy()                                                          
    ddict['cost_e'+str(i)]=allcostes[i].numpy()    
    ddict['loss'+str(i)]=alllosss[i].numpy()
    ddict['obj'+str(i)]=allobjs[i].numpy()    
    ddict['pkept'+str(i)]=allpkept[i].numpy()   
    ddict['preds'+str(i)] = allpreds[i].numpy()     
    ddict['sparse'+str(i)]=allsparse[i].numpy()        
    ddict['coherency'+str(i)]=allcoherent[i].numpy()  
    ddict['flex'+str(i)]=allflex[i].numpy()                  
    
    ddict['z'+str(i)] = allzs[i].numpy()
  for j in range(len(allcostovers)):
    ddict['over'+str(j)]=allcostovers[j].numpy()
  return(ddict)
  
    
    
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



def compute_loss_single(args,preds,y,zsum,zdiff):   
  if not args['binarize']:
    loss_mat = (preds-y)**2   
  else:
    if 'lsmooth' not in args or 'lsmooth'==0:
      loss_mat = tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=preds,
                                                      from_logits=0)
    else:
      print('\n\nLABEL SMOOTH')
      loss_mat = tf.keras.losses.binary_crossentropy(y_true=
                      tf.one_hot(tf.cast(y,dtype=tf.int32),depth=2),
                                                          y_pred=preds,
                                                      from_logits=0,
                                                      label_smoothing=args['lsmooth'])
  

  loss_vec = loss_mat
  loss =  tf.reduce_mean(input_tensor=loss_vec,axis=0)
  if 'scost' not in args or args['scost']=='L2':
    sparsity_metric = tf.reduce_mean((zsum-args['slevel'])**2)    
  elif args['scost']=='L1':
    sparsity_metric = tf.reduce_mean(tf.abs(zsum-args['slevel']))    
  coherent_metric = zdiff
  
  cost_g = loss + args['sparsity']*sparsity_metric + args['coherent']*coherent_metric
  cost_e = loss
  if args['binarize']:
    pred_hard = tf.cast(tf.equal(x=preds, y=tf.reduce_max(preds, -1, keepdims=True)),
                           y.dtype)
    pred_hard = pred_hard[:,1]
    right_or_wrong = tf.cast(tf.equal(x=pred_hard, y=y),
                           y.dtype)
    accuracy = tf.reduce_mean(right_or_wrong)
    
    obj = accuracy
  else:
    obj=cost_g
  return(cost_g,cost_e,loss,obj,sparsity_metric,coherent_metric)

