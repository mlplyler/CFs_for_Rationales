
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

sys.path.append('../fullcounter/')
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
  if len(args['pre_tran'])>0:
    if args['mtype']=='oememenc': #!!!!!!!!!!!!!!!!!!!!!!
      args['mtype']='oememencN' #!!!!!!!!!!!!!!!!!!!!!!
    #######
    # make the logpath
    if not os.path.exists(args['log_path']):
        os.makedirs(args['log_path'])
    # put logfile in right place
    from rationale_model import *        
    ## load default args
    args = add_default_args(args,rollrandom=True)
    args['logfile'] = args['log_path']+args['logfile']

    ## set random seed
    set_seeds(args['rand_seed'])
    # save the config file (argfile)
    with open(args['log_path']+'config.json','w') as f:
        json.dump(args,f,indent=2)
    
    ## make embedding   
    embed_layer = create_embedding_layer(args['embedding'],n_d=args['embdim'],v_max=args['n_v'])
    print('TRAIN FILE', args['train_file'])
    ## get data from file
    if args['binarize']:
      if 'split_rule' not in args or args['split_rule']=='poles':
        data_train = make_a_3gen(args['train_file'],args['aspects'],embed_layer,
                            args['max_len'],addstartend=0,binit=1)
        data_dev = make_a_3gen(args['dev_file'],args['aspects'],embed_layer,
                            args['max_len'],addstartend=0,binit=1)
      elif args['split_rule']=='poles_drop':
        data_train = make_a_3gen_drop(args['train_file'],args['aspects'],embed_layer,
                            args['max_len'],addstartend=0,binit=1,droprate=args['dropout'])
        data_dev = make_a_3gen(args['dev_file'],args['aspects'],embed_layer,
                            args['max_len'],addstartend=0,binit=1)
      elif args['split_rule']=='six':
        data_train = make_a_splitgen(args['train_file'],args['aspects'],embed_layer,
                            args['max_len'],addstartend=0,binit=1,splitnum=.6)
        data_dev = make_a_splitgen(args['dev_file'],args['aspects'],embed_layer,
                            args['max_len'],addstartend=0,binit=1,splitnum=.6)
    else:
        data_train = make_a_3gen(args['train_file'],args['aspects'],embed_layer,
                            args['max_len'],addstartend=0,binit=0)
        data_dev = make_a_3gen(args['dev_file'],args['aspects'],embed_layer,
                            args['max_len'],addstartend=0,binit=0)                              
    print('SPLIT RULE', args['split_rule'])
    ## shuffle training data                                            
    data_train = data_train.shuffle(10000,reshuffle_each_iteration=True)                      
    ## make and or load model
    amodel,theckpt,chkptman,ogen0,oenc0,ogen1,oenc1,ogen2,oenc2 = load_jrat(args,embed_layer,myModel)
    ## load cfd
    if 'fcbert' in args['pre_tran']:
      sys.path.append('../fcbert/')
      print('USING BERT')
      from cfmodel_bert import myModel as CF_Model
      from fcbert_utils import load_cf2
      cfd,args_cf = load_cf2(None,embed_layer=embed_layer,CF_Model=CF_Model,
                            chkptdir = args['pre_tran'],cftext='dummpy')  
      print('DONE LOAD')

    ## training loop stuff
    besdev_epoch=0
    ti_track=0
    thebesdev = np.inf
    gotsparse=False;firstsparse=False;
    TOTAKE=1
    for epoch in range(1):
      ## reset the logging dict
      tdict=defaultdict(list)
      devdict=defaultdict(list)
      ## training
      cgs=[];ces=[];losses=[];objs=[];pks=[];
      ti=0      
      trbsizes=[]
      print('TAKING BATCH')
      for by,bx in data_train.batch(args['train_batch']).take(TOTAKE):
        bsize=np.shape(bx)[0]
        print('BX', np.shape(bx), np.shape(by))
        if epoch==0 and ti==0:
          ## call both to set the tensor sizes
          cfdout = cfd['amodel'](bx,tar=bx,bsize=bsize,train=True)
          ratout = amodel(bx,train=False,bsize=bsize)
          
          if 'fcbert' in args['pre_tran']:
            print('USING BERT2')        
            amodel.encoders[0].tran.set_weights(cfd['amodel'].berty.encoder.get_weights())
            print('SET ENCODER')
            amodel.generator.glayers[0].set_weights(cfd['amodel'].berty.encoder.get_weights())   
            print('SET GENERATOR')
          else:
            amodel.encoders[0].tran.set_weights(cfd['amodel'].tran.encoder.get_weights())
            amodel.generator.glayers[0].set_weights(cfd['amodel'].tran.encoder.get_weights())   
        ratout = amodel(bx,train=False,bsize=bsize)
        print('DONE RATOUT')
        save_path = chkptman.save()
        print('saved bes to', save_path)
        print('EXITING CHKP SCRIPT')
        break
  else:
    print('NO PRE_TRAN, SKIPPED CHKP SCRIPT')
      
