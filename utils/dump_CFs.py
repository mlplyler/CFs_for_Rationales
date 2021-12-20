
import os
import pickle
import time
import json
import random

import numpy as np
import json
import argparse

import tensorflow as tf

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

    

def dump_loop(jpredict,elayer,thecfd):
  bsizes=[];     devdict=defaultdict(list)
  egs=[];ees=[];elosses=[];belosses=[];eobjs=[];epks=[];ezsum=[];ezdiff=[];
  newkeepers=[]
  print('dorandsamp', dorandsamp)
  print('flippit!!', flipit)
  
  with open(dumpfile,'a') as f:
    bi=0
    for ti,(by,bx_true,b_inds) in enumerate(data_train.batch(20).take(TOTAKE)):      
      bsize=np.shape(bx_true)[0]
      if bsize>0:
        bsizes.append(bsize)    
        t0 = time.time()
        if flipit:
          the_y = 1-by
        else:
          the_y = by
        ddict=jpredictby1_wrap(thefn=jpredict,args=args,jratd=jratd,
            cfd=thecfd,x=bx_true,
            y=the_y, 
            train=False,
            nounk=False,
            unkid=0,
            randsamp=dorandsamp,
            )
        ####################3
        ## grab old stuff that wasnt predicted...
        ddict['dec_ids']=ddict['dec_ids']*ddict['newz'] + ddict['newx']*(1-ddict['newz'])
        ########################
        ddict_cf=jpredicts2s_wrap_double(thefn=ratjdat,
                        args=args,
                        jratd=jratd,
                        cfd0=cfd0,
                        cfd1=cfd1,                      
                        x=ddict['dec_ids'],
                        y=ddict['newy'], 
                        train=False,  
                                                 
                        )     
        ddict['cf_pred']=  ddict_cf['newpred']  
        ddict['cf_z']=  ddict_cf['newz']                                                  
        dumpdicts = []
        
        for i in range(len(ddict['newy'])):
            dumpdicts.append({
                              'y':str(float(ddict['newy'][i])),
                              'pred':str(float(ddict['newpred'][i][1])),
                              'pred_cf':str(float(ddict['cf_pred'][i][1])),
                            'x':elayer.map_to_words(ddict['newx'][i]),
                            'cf':elayer.map_to_words([int(t) for t in ddict['dec_ids'][i]]),
                            'z':[str(float(zi)) for zi in ddict['newz'][i]],
                            'z_cf':[str(float(zi)) for zi in ddict['cf_z'][i]],
                            'b_ind':str(int(b_inds[i]))
                            })
 
        for j in range(len(dumpdicts)):
            json.dump(dumpdicts[j],f)
            f.write('\n') 
        bi+=1
    pflipped = 1-len(newkeepers)/np.sum(bsizes)
    print('PERCENT FLIPPED', pflipped)         
    print('num considered', np.sum(bsizes))
     
  return(devdict,bsizes,newkeepers,pflipped)    


#######################################
if __name__=='__main__':
  ######## load and parse args
  parser = argparse.ArgumentParser()
  parser.add_argument('thedir')
  parser.add_argument('dumpfile')
  parser.add_argument('-randsamp',default=False)
  parser.add_argument('-td',default='train')
  parser.add_argument('-flipit',default='1')
  parser.add_argument('-iterdecode',default='1')
  targs = parser.parse_args()  
  if targs.randsamp=='1' or targs.randsamp.lower=='true':
    print('doing rand samp',targs.randsamp)
    dorandsamp=True    
  else:
    print('doing ARGMAX', targs.randsamp)
    dorandsamp=False

  if targs.flipit=='1' or targs.flipit.lower=='true':
    print('FLIPPINGIT',targs.flipit)
    flipit=True    
  else:
    print('NOTTTT flippign it', targs.flipit)
    flipit=False

  

  dumpfile=targs.thedir+targs.dumpfile 
  ## args
  with open(targs.thedir+'/config.json','r') as f:
      cstr = f.read()
  args = json.loads(cstr)


  if targs.iterdecode=='1' or targs.iterdecode.lower=='true':
    print('ITERDECDE',targs.flipit)
    args['iterdecode']=1
  else:
    print('GREEDY DECODE', targs.flipit)
    args['iterdecode'] = 0    

  if targs.td=='train':
    thefile = args['train_file']
  elif targs.td=='dev':
    thefile = args['dev_file']
  else:
    raise NotImplemented

  print('THE FILE', thefile)

  if 1:
    from mratmodel_ind_tran_fix import myModel as Jrat_Model

    
    from cfmodel_bert import *
    from cfmodel_bert import myModel as CF_Model
    from Counterfactual_Predictor import *
    
    
  else:
    raise NotImplemented 
  
  ## load default args
  args = add_default_args(args,rollrandom=True)
  args['logfile'] = args['log_path']+args['logfile']
  args['cfboth_chkpt_dir']=targs.thedir #!!!!!!!!!!!!!!!!!!!!!!!!1


  ## set random seed
  set_seeds(args['rand_seed'])

  ## load models
  jratd,args = load_jrat(args,Jrat_Model=Jrat_Model)
  args = dict(args)
  cfd0,args = load_cf(args,CF_Model=CF_Model,embed_layer=jratd['elayer'],cftext='cfmodel0')
  args = dict(args)
  cfd1,args = load_cf(args,CF_Model=CF_Model,embed_layer=jratd['elayer'],cftext='cfmodel1')  
  args = dict(args)

  print('NOUNKID', cfd0['elayer'].vocab_map['<unk>'])

  
  ## training loop stuff
  besdev_epoch=0
  thebesdev = np.inf
  gotsparse=False;firstsparse=False;
  
  
  if 'TOTAKE' in args:
     TOTAKE=args['TOTAKE']
  else:
    TOTAKE=100000
  for toclass in [0,1]:
    thecfd = cfd0 if toclass==0 else cfd1
    args['slevel']=.10 #!!!!
    keepers=list(range(100000))
    print('\n\n\n')
    print('SLEVEL',args['slevel'], toclass)
    print('\n')
    if flipit:
      the_toclass = 1-toclass
    else:
      the_toclass = toclass
    data_train = make_a_fcgen_sorted_track(
                  thefile,args['aspects'],jratd['elayer'],
                      args['max_len'],addstartend=1,binit=1,
                      classpick=the_toclass,##!!
                      keepind=keepers)
    devdict,bsizes,keepers,pflipped=dump_loop(jpredict=jpredict_by1,
                            elayer=jratd['elayer'],thecfd=thecfd)
    print('num keepers', len(keepers))
