
import os
import pickle
import time
import json
import random
import math


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
from collections import Counter
from scipy.stats import entropy
#######################################
def update_ldict(ldict,ddict):
  for k in ddict:
    ldict[k].append(ddict[k])
  return(ldict)

def get_flipacc(toclass,pred_cf): 
  pred_bin = tf.cast(tf.math.greater_equal(pred_cf[:,1],.5),dtype=tf.int32) 
  if toclass=='positive':
    to_y = tf.ones(tf.shape(pred_bin),dtype=tf.int32)
  else:
    to_y = tf.zeros(tf.shape(pred_bin),dtype=tf.int32)
  eqs = tf.math.equal(to_y,pred_bin)
  acc = tf.reduce_sum(tf.cast(eqs,dtype=tf.int32),axis=-1)/tf.shape(pred_cf)[0]
  
  return(acc)

def update_fillcounter(xcf,zs,fillcounter):
  xcf = xcf.numpy()
  zs = zs.numpy()
  for x,z in zip(xcf,zs):
    arep = [int(ti) for ti,zi in zip(x,z) if zi>=.5]
    fillcounter.update(arep)
  return(fillcounter)

def calc_ent_from_counts(fillcounter):
  '''
  my understanding of scipy.stats.entropy is that it will ignore zero probability
  so entropy([.33,.33,.33,0,0,0,0])=entropy([.33,.33,.33])
  this function does the later
  '''
  p0 = np.array(list(fillcounter.values()))
  p0 = p0/np.sum(p0)
  ent = entropy(p0)
  return(ent)



def flip_est(args,cfmodel,ratmodel,data_gener,train):    
  accs=[];bsizes=[]; fillcounter=Counter();
  for ii,(y,x) in enumerate(data_gener.batch(50).take(10)):    
    bsize=np.shape(x)[0]
    bsizes.append(bsize)
    cost_d,x_nose,allpreds,z,x_nose_cf,allpreds_cf,z_cf = jpredict_wdat(
                           args,ratmodel,cfmodel,x,y,train,bsize=bsize)
    accs.append(get_flipacc(args['toclass'],allpreds_cf[0]))
    fillcounter = update_fillcounter(x_nose_cf,z,fillcounter)
  flipent = calc_ent_from_counts(fillcounter)
  flipacc = np.dot(accs,bsizes)/np.sum(bsizes)
  return(flipacc,flipent)                           
################################################
def dump_it(epoch,dumpfile,args,cfmodel,ratmodel,x,y,train,bsize):
  ## this is not flipped
  cost_d,x_nose,allpreds,z,x_nose_cf,allpreds_cf,z_cf = jpredict_wdat(
                           args,ratmodel,cfmodel,x,y,train,bsize=bsize)
  x_nose=x_nose.numpy()                           
  allpreds=allpreds[0].numpy()
  z = z.numpy()
  x_nose_cf=x_nose_cf.numpy()
  allpreds_cf=allpreds_cf[0].numpy()
  z_cf = z_cf.numpy()
  y=y.numpy()
  

  with open(dumpfile,'a') as f:
    for i in range(len(x_nose)):
      dumpdict = {
                  'y':str(float(y[i])),
                  'pred':str(float(allpreds[i][1])),
                  'pred_cf':str(float(allpreds_cf[i][1])),              
                'x':cfd['elayer'].map_to_words([int(t) for t in x[i]]),
                'cf':cfd['elayer'].map_to_words([int(t) for t in x_nose_cf[i]]),
                'z':[str(float(zi)) for zi in z[i]],
                'z_cf':[str(float(zi)) for zi in z_cf[i]],   
                'rowid':str(epoch)+'__'+str(i)           
                }
      json.dump(dumpdict,f)
      f.write('\n')
    
#######################################
def GAN_wrap(args,thefn,cfd,jratd,discer,opt_discer,
            x,y,bsize,
             train,ganlambda):
  (cfloss,cfpredloss,ganloss) = thefn(args,cfd['amodel'],
                                    discer,jratd['amodel'],
                                    x,y,bsize,
                                    cfd['opter'],opt_discer,
                                    train,ganlambda)  
  ddict = {
        'cfloss':cfloss.numpy(),
        'cfpredloss':cfpredloss.numpy(),
        'ganloss':ganloss.numpy(),        
            }   
  return(ddict)
    
    
def training_loop(cag,thebsize=None):
  max_dev_obj=-1
  NEVERMET=1
  trbsizes=[];    tdict=defaultdict(list);
  if thebsize is None:
    thebsize=args['train_batch']
  tii=0
  for epoch in range(args['epochs_js2s']): #######!!!!!!!!!!!
    print('EPOCH JS2S', epoch)    
    for ti,(by,bx) in enumerate(data_train.batch(thebsize).take(TOTAKE)):
      if epoch==0 and ti<args['GANwarmsteps']:
        print('WARMUP', epoch, ti)
        ganlambda=float(args['GANlambda'])/10
        the_cag = cag_JD
      else:
        
        ganlambda=float(args['GANlambda'])
        the_cag = cag
        
      bsize=np.shape(bx)[0]
      
      if ti==0 and dodump==1:
        dump_it(epoch,args['log_path']+args['toclass']+'.dump',
                  args,cfd['amodel'],jratd['amodel'],bx,by,
                  train=False,bsize=bsize)  
        print('DONE DUMP')
      if bsize>1:      
        trbsizes.append(bsize)
        ddict=GAN_wrap(thefn=the_cag,
                        args=args,                      
                        cfd=cfd,                      
                        jratd=jratd,
                        discer=discer,
                        opt_discer=opt_discer,
                        x=bx,
                        y=by,
                        bsize=bsize,
                        train=True,
                        ganlambda=ganlambda                             
                        )
        
        tdict = update_ldict(tdict,ddict)
      
        if tii%50==0:
          print('by', np.mean(by.numpy()),np.shape(by))      
          printnsay(thefile=args['logfile'],
              text = 'traini:'+str(tii)+','+','.join(['{}:{:.5f}'.format(x,ddict[x]) for x in ddict]))
        else:
          justsay(thefile=args['logfile'],
              text = 'traini:'+str(tii)+','+','.join(['{}:{:.5f}'.format(x,ddict[x]) for x in ddict]))
        if tii%CHECKNUM==0:
          flipacc,flipent = flip_est(args,cfmodel=cfd['amodel'],
                    ratmodel=jratd['amodel'],data_gener=freshgen(args),
                    train=False)    


          printnsay(thefile=args['logfile'],
                        text = 'epoch:{:.0f}'.format(epoch) + ','
                        +','.join(['train_{}:{:.5f}'.format(x,np.dot(trbsizes,tdict[x])/np.sum(trbsizes)) 
                                                      for x in tdict]) +','
                        #+','.join(['dev_{}:{:.5f}'.format(x,np.dot(bsizes,devdict[x])/np.sum(bsizes)) 
                        #                              for x in devdict])
                        
                        ) 
          print('flipacc', np.shape(flipacc), 'flipent', np.shape(flipent))
          dev_obj = args["chk_obj_mult"]*flipacc+flipent ## args[] wasnt on for 0 and 100 ex2 gan but same as default of 1
          if dev_obj>max_dev_obj:
            cond=True
            max_dev_obj=dev_obj
          else:
            cond=False

          if  cond: 
            print('CHECKPOINTING BABY')
            printnsay(thefile=args['logfile'],
              text = 'chkpting BABEE,  {:.5f}, {:.5f},{:.5f}'.format(
                              flipacc,flipent,dev_obj      
              ))            
            chkpt_it(args,cfd)               
          else:
            printnsay(thefile=args['logfile'],
              text = 'skip chkpt,  {:.5f}, {:.5f}'.format(
                              flipacc,flipent      
              ))
          ## RESET STUFF!!!
          trbsizes=[];    tdict=defaultdict(list);
        tii+=1          




      #ti_track+=1
  return(tdict,trbsizes)

def chkpt_it(args,modeld):#ratd,cfd):
  modeld['theckpt'].step.assign_add(1)
  save_path = modeld['chkptman'].save()
  print('saved bes to', save_path)
  printnsay(thefile=args['logfile'],text = 'saved chkpt '+str(modeld['theckpt'].step))
      


#######################################
if __name__=='__main__':
  ######## load and parse args
  parser = argparse.ArgumentParser()
  parser.add_argument('configs')
  targs = parser.parse_args()  
  #@title setup args
  ## args
  with open(targs.configs,'r') as f:
      cstr = f.read()
  args = json.loads(cstr)

  # make the logpath
  time.sleep(random.randrange(1,5))
  if not os.path.exists(args['log_path']):
      os.makedirs(args['log_path'])  

  
  

  from rationale_model import myModel as Jrat_Model
  from rationale_model import Encoder_tran
  
  
  
  from cfmodel_bert import *
  from cfmodel_bert import myModel as CF_Model
  from fc_bert2 import *
  
    
    
  args = add_default_args(args,rollrandom=True)
 
  if args['toclass']=='positive':
    cftext='cfmodel1'
  else:
    cftext='cfmodel0'   
  args['logfile'] = args['log_path']+cftext+'_'+args['logfile']

  ## set random seed
  set_seeds(args['rand_seed'])  
  
  ## load models

  if 'new_SL' in args:
    args['slevel'] = args['new_SL']

  jratd,args = load_jrat(args,Jrat_Model=Jrat_Model)
  cfd,args = load_cf(args,CF_Model=CF_Model,embed_layer=jratd['elayer'],cftext=cftext,
                      dub_opt=True)
  
  print('\n\n\n\n SLEVEL', 
      args['slevel'],
      jratd['amodel'].args['slevel'],
      cfd['amodel'].args['slevel'],
      '\n\n\n')

  # ## make the discriminator!!
  data_train_dumb = make_a_fcgen(args['train_file'],args['aspects'],jratd['elayer'],
                       args['max_len'],addstartend=1,binit=1,classpick=args['classpick'])
  
  discer_enc = Encoder_tran(args,jratd['elayer'])  
  if args['tgan']:
    discer = TDiscer(args,discer_enc.tran,discer_enc.padding_id)
  else:  
    discer = Discer(args,discer_enc)

  for ti,(by,bx) in enumerate(data_train_dumb.batch(args['train_batch']).take(1)):
    x_nose = remove_start_end(bx,cfd['amodel'].start_id,cfd['amodel'].end_id,
                                cfd['amodel'].padding_id)                                                                
    ## initialize from jratd                            
  
    masks = tf.cast(tf.not_equal(x_nose, jratd['amodel'].generator.padding_id),
                        tf.float32,
                        name = 'masks_generator')
    x_nose2 = tf.one_hot(x_nose,depth=args['n_v']) 
    bsize=np.shape(x_nose)[0]                       
    moo  = discer(x_nose,masks,from_logits=False,bsize=bsize)

    ## initialize from cfd
    moo = cfd['amodel'](bx,bx,np.shape(bx)[0],train=True)

  ## initialize from jratd
  
  ## initialize from cfd
  if args['tgan']:
    discer.tran.set_weights(cfd['amodel'].berty.encoder.get_weights())
  else:
    discer.encoder.tran.set_weights(cfd['amodel'].berty.encoder.get_weights())

  opt_discer = cfd['opter2'] #!!!!!!!!!!!!!!!!!!

  numlines = sum(1 for line in open(args['train_file']))
  numbatch = math.ceil(numlines/args['train_batch'])
  if numbatch<args['checknum']:
    CHECKNUM=numbatch
  else:
    CHECKNUM=args['checknum']
  if CHECKNUM>args['TOTAKE']:
    CHECKNUM=args['TOTAKE']
  print('CHECKNUM', CHECKNUM)
  ## get data generators
  data_train = make_a_fcgen(args['train_file'],args['aspects'],jratd['elayer'],
                      args['max_len'],addstartend=1,binit=1,
                      classpick=args['classpick'])

  freshgen = lambda args: make_a_fcgen(args['train_file'],args['aspects'],jratd['elayer'],
                      args['max_len'],addstartend=1,binit=1,
                      classpick=1-args['toint']) ## want the source to be opposite the toint

  
  ## training loop stuff
  besdev_epoch1=0  
  thebesdev1 = np.inf
  gotsparse1=False;firstsparse1=False;
  

  besdev_epoch0=0
  thebesdev0 = np.inf
  gotsparse0=False;firstsparse0=False;
  
  if 'TOTAKE' in args:
     TOTAKE=args['TOTAKE']
  else:
    TOTAKE=100000

  # save the config file (argfile)
  with open(args['log_path']+'config.json','w') as f:
      json.dump(args,f,indent=2)     
  with open(args['log_path']+args['toclass']+'_config.json','w') as f:
      json.dump(args,f,indent=2)           

  
  dodump=1

  cag = single_GANbyc #!!!!!!!!!!
  cag_JD=disceronly_GAN
  oganlambda=float(args['GANlambda'])




  tdict,trbsizes=training_loop(cag=cag,thebsize=args['train_batch'])
  printnsay(thefile=args['logfile'],text = 'DONE!!!')




