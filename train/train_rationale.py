
import os
import pickle
import time
import json

import numpy as np
import json
import argparse

import tensorflow as tf

import sys
sys.path.append('../share/')
from IO import *
from embeddings_novector import *

from ratstuff import set_seeds
#######################################
def update_ldict(ldict,ddict):
  for k in ddict:
    ldict[k].append(ddict[k])
  return(ldict)
#############################################
#############################################
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('configs')
  targs = parser.parse_args()  
  ## args
  with open(targs.configs,'r') as f:
      cstr = f.read()
  args = json.loads(cstr)
  #######
  # make the logpath
  if not os.path.exists(args['log_path']):
      os.makedirs(args['log_path'])
  # put logfile in right place

  from rationale_model import *        

  ## load default args
  args = add_default_args(args,rollrandom=True)
  if args['logfile']=='logfile.log': 
    args['logfile'] = args['log_path']+args['logfile']

  
  ## set random seed
  set_seeds(args['rand_seed'])  

  ## make embedding   
  embed_layer = create_embedding_layer(args['embedding'],n_d=args['embdim'],v_max=args['n_v'])

  print('TRAIN FILE', args['train_file'])
  ## get data from file

  if 'split_rule' not in args or args['split_rule']=='poles':
      data_train = make_a_3gen(args['train_file'],args['aspects'],embed_layer,
                          args['max_len'],addstartend=0,binit=1)
      data_dev = make_a_3gen(args['dev_file'],args['aspects'],embed_layer,
                          args['max_len'],addstartend=0,binit=1)
                          
  print('SPLIT RULE', args['split_rule'])
  ## shuffle training data                                            
  data_train = data_train.shuffle(10000,reshuffle_each_iteration=True)                      
  ## make and or load model
  if len(args['pre_tran'])>0:
    amodel,theckpt,chkptman,ogen0,oenc0,ogen1,oenc1,ogen2,oenc2 = load_jrat(args,
                  embed_layer,myModel,chkptdir=args['log_path'])
  else:
    print('PRE_TRAN is empty, not loading any checkpoint')
    amodel,theckpt,chkptman,ogen0,oenc0,ogen1,oenc1,ogen2,oenc2 = load_jrat(args,
                  embed_layer,myModel)
                
  ## training loop stuff
  besdev_epoch=0
  ti_track=0
  thebesdev = np.inf
  gotsparse=False;firstsparse=False;
  
  if 'TOTAKE' in args:
    TOTAKE=args['TOTAKE']
  else:
    TOTAKE=400000000



  with open(args['log_path']+'config.json','w') as f:
      json.dump(args,f,indent=2)
  
  
  for epoch in range(args['TOTAKE']):
    ## reset the logging dict
    tdict=defaultdict(list)
    devdict=defaultdict(list)
    ## training
    cgs=[];ces=[];losses=[];objs=[];pks=[];
    ti=0
    
    trbsizes=[]
    for by,bx in data_train.batch(args['train_batch']).take(TOTAKE):
      bsize=np.shape(bx)[0]
      trbsizes.append(bsize)

      ddict=cag_wrap_fix(compute_apply_gradients=compute_apply_gradients,
                      args=args,
                      model=amodel,
                      x=bx,
                      y=by,
                      optimizer_gen0=ogen0,
                      optimizer_enc0=oenc0,                      
                      train=tf.constant(True,dtype=tf.bool),  ## should change this!!
                      bsize=bsize
                      )
      tdict = update_ldict(tdict,ddict)

      if ti%25==0:
        print(epoch,',', ti)
        printnsay(thefile=args['logfile'],
          text = 'traini:'+','+','.join(['{}:{:.5f}'.format(x,ddict[x]) for x in ddict]))
      else:
        justsay(thefile=args['logfile'],
          text = 'traini:'+','+','.join(['{}:{:.5f}'.format(x,ddict[x]) for x in ddict]))
                              
      ti+=1
      ti_track+=1
    if 1:
        print('EVAL', ti_track)
        ## evaluation
        bsizes=[]
        egs=[];ees=[];elosses=[];belosses=[];eobjs=[];epks=[];ezsum=[];ezdiff=[];
        for by,bx in data_dev.batch(args['eval_batch']).take(TOTAKE):
          
          bsize = np.shape(bx)[0]
          bsizes.append(bsize)          
          dev_ddict=just_predict_fix(jpraw=jpraw,args=args,
                                          model=amodel,
                                          x=bx,
                                          y=by,
                                          train=tf.constant(False,dtype=tf.bool),
                                          bsize=bsize)        
                                        
          devdict = update_ldict(devdict,dev_ddict)
        ## log it

        printnsay(thefile=args['logfile'],
                  text = 'epoch:{:.0f}'.format(epoch) + ','
                  +','.join(['train_{}:{:.5f}'.format(x,np.dot(trbsizes,tdict[x])/np.sum(trbsizes)) 
                                                for x in tdict]) +','
                  +','.join(['dev_{}:{:.5f}'.format(x,np.dot(bsizes,devdict[x])/np.sum(bsizes)) 
                                                for x in devdict])) 
                  

        devdeviation = np.max(np.abs([np.dot(bsizes,devdict[x])/np.sum(bsizes) 
                            -args['wanted_sparse']
                            for x in devdict if 'pkept' in x]))
        print('devdeviation', devdeviation)

        dev_obj = np.mean([np.dot(bsizes,devdict[x])/np.sum(bsizes)
                           for x in devdict if 'cost_g' in x]) ## gen cost
        

        
        if devdeviation<=args['sparse_margin']:
          if not gotsparse:
            firstsparse=True
          else:
            firstsparse=False ## only flips to false if its sparse and previously sparse
          gotsparse=1 ## gotsparse is never false after becoming true
          
        thetolerance = 0          
        if (thebesdev-dev_obj)>thetolerance:
          betteracc=1
        else:
          betteracc=0

        if (
           (gotsparse and firstsparse)    or
           (gotsparse and betteracc)  or
           (not gotsparse and betteracc)
            ):  


          besdev_epoch = epoch
          thebesdev = dev_obj

          print('NEW BEST!!', thebesdev)
          if args['dosave']:
            
            save_path = chkptman.save()
            print('saved bes to', save_path)
            printnsay(thefile=args['logfile'],text = 'saved chkpt '+str(epoch))
        if epoch>=args['abs_max_epoch']:
          printnsay(thefile = args['logfile'],
                  text = 'BROKE!! epoch '+str(epoch))
          break
        ## track the epoch
        theckpt.step.assign_add(1)
