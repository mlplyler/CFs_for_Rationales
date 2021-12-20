
import os
import pickle
import time
import json
import random

import numpy as np
import json
import argparse

import tensorflow as tf
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

import sys
sys.path.append('../share/')
from IO import *
from embeddings_tokrep import *
from fcbert_utils import *
from ratstuff import set_seeds
#######################################
def update_ldict(ldict,ddict):
  for k in ddict:
    ldict[k].append(ddict[k])
  return(ldict)
#######################################

def cf_single_wrap(thefn,args,cfd,x,y,mybmask,train=True):
  bsize=np.shape(x)[0]
 
  dec_ids,costd = thefn(args=args,cfmodel=cfd['amodel'],x_in=x,x_tar=y,
          mybmask=mybmask,bsize=bsize,opt_cf=cfd['opter'],train=train)
                                          
  ddict={
        'costd':costd.numpy(),
        }
  return(ddict)      
  

def eval_loop():
  bsizes=[];     devdict=defaultdict(list)
  egs=[];ees=[];elosses=[];belosses=[];eobjs=[];epks=[];ezsum=[];ezdiff=[];
  for bx_m,bx_true,mybmask in data_dev.batch(args['eval_batch']).take(TOTAKE):
    bsize=np.shape(bx_m)[0]
    if bsize>1:
      bsizes.append(bsize)
    
      ddict=cf_single_wrap(thefn=jpredict_js2s_single_wmask,
                      args=args,
                      cfd=cfd,
                      x=bx_m,
                      y=bx_true,
                      mybmask=mybmask,                      
                      train=False                                 
                      )       
      devdict = update_ldict(devdict,ddict)
  return(devdict,bsizes)    

def chkpt_it(args,modeld):
  modeld['theckpt'].step.assign_add(1)
  save_path = modeld['chkptman'].save()
  print('saved bes to', save_path)  

def chkpt_logic(besdev_epoch,thebesdev,gotsparse,firstsparse,cfd,
                    n_to_break=15,node=0):

  dev_obj = np.mean([np.dot(bsizes,devdict[x])/np.sum(bsizes)
                     for x in devdict if 'costd' in x and 'cf' not in x]) ## gen cost
  
  print('dev_objjjjjjj', dev_obj)
  
  if dev_obj<thebesdev:

    besdev_epoch = epoch
    thebesdev = dev_obj

    print('NEW BEST!!', thebesdev)
    if args['dosave']:
      chkpt_it(args,cfd)            

  if epoch>args['epochs_js2s'] and (epoch-besdev_epoch)>n_to_break:
    
    printnsay(thefile = args['logfile'],
            text = 'BROKE!! epoch '+str(epoch)+' '+str(epoch-besdev_epoch))
    return(0,besdev_epoch,thebesdev,gotsparse,firstsparse)
  else:
    return(1,besdev_epoch,thebesdev,gotsparse,firstsparse )





#######################################
if __name__=='__main__':

  with strategy.scope():
    print('STRATEGY', strategy)    
    print('tf config', os.environ['TF_CONFIG'])
    tf_config = json.loads(os.environ['TF_CONFIG'])
    print('tf_config', tf_config)
    ######## load and parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('configs')
    targs = parser.parse_args()  
    ## args
    with open(targs.configs,'r') as f:
        cstr = f.read()
    args = json.loads(cstr)
    # make the logpath
    if not os.path.exists(args['log_path']):
        try:
          os.makedirs(args['log_path'])  
        except:
          print('DONE GOT LOG_PATH')

    
    from cfmodel_bert import *
    from cfmodel_bert import myModel as CF_Model
  
    
    ## load default args
    args = add_default_args(args,rollrandom=True)
    args['logfile'] = args['log_path']+args['logfile']

    ## set random seed
    set_seeds(args['rand_seed'])
    
 
        
    ## load models
    cfd,args = load_cf(args,CF_Model=CF_Model,
                cftext='cfmodel'+str(tf_config['task']['index']),logfile=args['logfile'])    
    ## get data generators
    data_train =  make_a_maskgen_jcont(args['train_file'],args['aspects'],cfd['elayer'],
                        args['max_len'],
                        addstartend=1,binit=1,perchigh=args['slevel'],
                        classpick=args['classpick'])
    data_train = data_train.shard(num_shards=len(tf_config['cluster']['worker']),
                                  index = tf_config['task']['index'])                      
    data_dev =  make_a_maskgen_jcont(args['dev_file'],args['aspects'],cfd['elayer'],
                        args['max_len'],
                        addstartend=1,binit=1,perchigh=args['slevel'],
                        classpick=args['classpick'])
    ## training loop stuff
    besdev_epoch=0
    thebesdev = np.inf
    gotsparse=False;firstsparse=False;
    ## training loop stuff
    besdev_epoch=0
    ## do the training, all on 
    if 'TOTAKE' not in args:
      TOTAKE=10000000
    else:
      TOTAKE=args['TOTAKE']
    # save the config file (argfile)
    with open(args['log_path']+'config.json','w') as f:
        json.dump(args,f,indent=2)       
    
    @tf.function
    def distributed_train_step(strategy,args_,cfmodel,x_in,x_tar,mybmask,bsize,opt_cf,train,node):
      strategy.experimental_run_v2(cag_js2s_single_wmask_L2,args=(args_,cfmodel,x_in,x_tar,mybmask,bsize,
                            opt_cf,True))                           
    break_status=1                     
    for epoch in range(1000+args['epochs_js2s']):

      print('EPOCH', epoch)
      t0=time.time()      
      for ti,(bx_m,bx_true,mybmask) in enumerate(data_train.batch(args['train_batch'],
                                                    drop_remainder=True).take(TOTAKE)):
        bsize=tf.shape(bx_m)[0]
        print('ti', ti, bsize)
        distributed_train_step(strategy,args,cfd['amodel'],bx_m,bx_true,mybmask,
                                bsize,cfd['opter'],train=True,
                                node=tf_config['task']['index'])
      t1 = time.time()-t0    
      print('EPOCH TRAIN TIME', t1)   
      devdict,bsizes=eval_loop()      
      if 1:
        printnsay(thefile=args['logfile'],
            text = 'epoch:{:.0f}'.format(epoch) + ', time:{:.3f}'.format(t1)+\
                                ', node:{:.3f}'.format(tf_config['task']['index'])+',' +\
                    ','.join(['dev_{}:{:.5f}'.format(x,np.dot(bsizes,devdict[x])/np.sum(bsizes)) 
                                                  for x in devdict])) 

      break_status,besdev_epoch,thebesdev,gotsparse,firstsparse = chkpt_logic(
                                                besdev_epoch,thebesdev,gotsparse,firstsparse,
                                                          cfd,node=tf_config['task']['index'])
      if not break_status:
        break
      
      
    

