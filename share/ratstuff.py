import tensorflow as tf
import numpy as np

import random
import os

#################################################################
def set_seeds(theseed):
  np.random.seed(theseed)
  tf.random.set_seed(theseed)  
  random.seed(theseed)
  os.environ['PYTHONHASHSEED']=str(theseed)


##################################################################
dloss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,reduction='none')

def compute_loss_dec1(args,dscore,preds2,h_og_content,h_new_content):   
  if not args['binarize']:  
    style_cost = tf.reduce_mean(tf.keras.losses.MSE(dscore,preds2))
  else:
    style_cost= tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true=dscore,y_pred=preds2,
                                                      from_logits=0))
 
  content_cost =  tf.reduce_mean(tf.keras.losses.MSE(h_og_content,h_new_content))
  cost_d = args['style']*style_cost+args['content']*content_cost

  return(cost_d,style_cost,content_cost)



def compute_loss_gen_enc(args,preds,y,zsum,zdiff): 
  y=tf.squeeze(y)
  preds = tf.squeeze(preds)
  if not args['binarize']:
    loss_mat = (preds-y)**2   
  else:    
    loss_mat = tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=preds,
                                                      from_logits=0)
  loss_vec = loss_mat
  loss =  tf.reduce_mean(input_tensor=loss_vec,axis=0)
  if 'scost' not in args or args['scost']=='L2':    
    sparsity_cost = args['sparsity']*tf.reduce_mean((zsum-args['slevel'])**2) +\
                    args['coherent']*zdiff 
  elif args['scost']=='L1':
    sparsity_cost = args['sparsity']*tf.reduce_mean(tf.abs(zsum-args['slevel'])) +\
                  args['coherent']*zdiff

  
  cost_g = sparsity_cost + loss
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
  return(cost_g,cost_e,loss,obj)



def compute_loss_dec(args,
                 logits,xtrue,pad_id):   
  cost_d=0
  
  ## decoder loss
  xtrue =xtrue[:,1:]
  logits=  logits[:,1:]
  dloss = dloss_fn(xtrue,logits)
  ## apply mask
  if 'mask_pad' in args and args['mask_pad']:
    print('MASK PAD IS ON!!!!!')
    mask = tf.cast(tf.not_equal(xtrue, pad_id),
                          tf.float32)
    mask = tf.cast(mask,dtype=dloss.dtype)
    dloss = dloss*mask
  cost_d = tf.reduce_mean(dloss)
  ############
  return(cost_d)



##################################
### OUT LAYER ###################      
##################################
class Outlayer(tf.keras.Model):
  def __init__(self,args):
    super().__init__()
    self.args = args
    if args['binarize']:
      nclass = args['numclass']*2
    else:
      nclass=args['numclass']    
    if 'outact' not in args or args['outact']=='sigmoid':
      print('\n\n OUTLAYER \n\n', 'sigmoid')  
      self.outlayer = tf.keras.layers.Dense(nclass,activation='sigmoid')   ########## UMMMM SHIIIIT
    elif args['outact']=='softmax':
      print('\n\n OUTLAYER \n\n', 'softmax')  
      self.outlayer = tf.keras.layers.Dense(nclass,activation='softmax')   ########## UMMMM SHIIIIT                    
  def call(self,x):
    preds = self.outlayer(x)
    return(preds)
  def getL2loss(self,):
      # get l2 cost for all parameters
      lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in self.trainable_variables
                   if 'bias' not in v.name.lower() ]) * self.args['l2_reg']
      return(lossL2)























