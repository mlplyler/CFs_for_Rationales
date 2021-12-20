
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
sys.path.append('../mrat/')
sys.path.append('../s2sonly/')
sys.path.append('../enconly/')
from IO import *
from mrat_utils import *
from sklearn.metrics import precision_score, recall_score, f1_score
from ratstuff import set_seeds
from collections import Counter

##############################
##############################
#############################################
#############################################


#######################################
def update_ldict(ldict,ddict):
  for k in ddict:
    ldict[k].append(ddict[k])
  return(ldict)


def get_prec(bx,z,tz,padid):
  znew=[]
  tznew=[]
  for i in range(len(z)):
    padinds = np.where(bx[i]==padid)[0]
    if len(padinds)>0:
      minpadind = np.min(padinds)
      znew.append(z[i,:minpadind])
      tznew.append(tz[i,:minpadind])
    else:
      znew.append(z[i])
      tznew.append(tz[i])

  precs = [precision_score(np.round(tznew[i]),np.round(znew[i])) 
            for i in range(len(znew))]  
  prec1 = np.mean(precs)

  f1s = [f1_score(np.round(tznew[i]),np.round(znew[i])) 
            for i in range(len(znew))]  
  f1s1 = np.mean(f1s)

  recalls = [recall_score(np.round(tznew[i]),np.round(znew[i])) 
            for i in range(len(znew))]  
  recall1 = np.mean(recalls)


  tznewf = [];znewf=[];
  for tzi,zi in zip(tznew,znew):
    tznewf+=list(tzi)
    znewf+=list(zi)
  prec2 = precision_score(np.round(tznewf), 
                          np.round(znewf))
  recall2 = recall_score(np.round(tznewf), 
                          np.round(znewf))
  f1s2 = precision_score(np.round(tznewf), 
                          np.round(znewf))                          
  zlen = len(znewf)
  return(prec1,prec2,
         recall1,recall2,
         f1s1,f1s2,
          zlen)
  


def remove_padding_ids(anparray,padding_id):
  masked = np.ma.masked_equal(anparray,padding_id)
  return(masked.compressed())


def fix_startend(strlist):
  if strlist[0]=='<start>':
    strlist = strlist[1:]
    strlist.append('<padding>')
  if '<end>' in strlist:
    theind = strlist.index('<end>')
    strlist[theind] = '<padding>'
  return(strlist)
  
def zwcounts(ddict,x,usettrack):

  z = ddict['z0']
  
  xzmasked = np.ma.masked_array(x,mask=z)
  vals = xzmasked[xzmasked.mask==True].data
  
  u,c = np.unique(vals,return_counts=True)
  u = [str(ui) for ui in u]
  c = [int(ci) for ci in c]
  acount =   Counter(dict(zip(u, c)))
  usettrack = sum([usettrack,acount],Counter())
  return(usettrack)

    

##########################################
#########################################
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('args')
  parser.add_argument('ft',default=0)
  targs = parser.parse_args()
  

  with open(targs.args,'r') as f:
        cstr = f.read()

        
  args = json.loads(cstr)
  if 'rand_seed' in args:
    set_seeds(args['rand_seed'])
    print('GOT RAND SEEED!!!!!!!!!!!!')
  else:
    print('NO RANDOM SEED')


  ## LOAD RATMODEL
  # make the model
  from rationale_model import *
  args = add_default_args(args,rollrandom=True)


  


  ###############################################################################
  ## make embedding   
  
  ## make embedding   
  if 'home' in args['embedding']:
    args['embedding'] = '/mnt/beegfs/mlplyler/embeds/'+args['embedding'].split('/')[-1]

  if 'ind' not in args['mtype']:
    from embeddings_novector import * 
    embed_layer = create_embedding_layer(args['embedding'],n_d=args['embdim'],v_max=args['n_v'])        
  else:
    from embeddings_tokrep import *
    embed_layer = create_embedding_layer(args['embedding'],n_d=args['embdim'])
  print('ARGS EMBEDDING', args['embedding'])
  #####################################################################


 #############################################
  #############################################

  #############################################      
  
  ## get data from file
  if args['binarize']:
    if 'json' in args['source_file']:
        data_dev = make_a_3gen(args['dev_file'],args['aspects'],embed_layer,
                            args['max_len'],addstartend=0,binit=1)
        data_finedev = make_a_3gen(args['finedev_file'],args['aspects'],embed_layer,
                            args['max_len'],addstartend=0,binit=1)                            
        dataset = make_a_jsongen(args['source_file'],args['aspects'],
                          embed_layer,maxlen=args['max_len'],
                      addstartend=0,binit=0)                              
                      
          
    else:
      data_dev = make_a_3gen(args['dev_file'],args['aspects'],embed_layer,
                            args['max_len'],addstartend=0,binit=1)
      data_finedev = make_a_3gen(args['finedev_file'],args['aspects'],embed_layer,
                            args['max_len'],addstartend=0,binit=1)
      dataset = make_a_hotelgen(args['source_file'],args['aspects'],
                          embed_layer,maxlen=args['max_len'],
                      addstartend=0,binit=1)                          
  print('DATASET', dataset, type(dataset))
  #####################################################################




  if 'mtype' not in args or args['mtype']!='fc2':
    
    if targs.ft=='1':
      args['res_file']+='.FT'
      args['load_chkpt_dir']=args['log_path']+'/finetuned/' 
      args['log_path'] = args['log_path']+'/finetuned/' 
    else:
      args['load_chkpt_dir']=args['log_path']
    amodel,_,_,_,_,_,_,_,_= load_jrat(args,embed_layer,myModel)
  else:
    jratd = load_jrat(args,Jrat_Model=Jrat_Model)
    amodel = jratd['amodel']   
  

  #####################################################################
  results={}
  results['args'] = dict(args)
  results['time']=time.strftime('%X %x %Z')
  ## get the chkptfile
  cname = args['log_path']+ [f for f in os.listdir(args['log_path'])
                                       if 'ckpt-' in f and '.data' in f][0]
  results['chkpt'] = cname
  
  
  results['dev']={};results['test']={};results['finedev']={};
  tdict=defaultdict(list)
  devdict=defaultdict(list)
  finedevdict=defaultdict(list)
  
  if len(args['savename'])>0:
    savepath = '/'.join(args['savename'].split('/')[:-1])
    if not os.path.exists(savepath):
      os.makedirs(savepath)
  
  
  ###############
  if os.path.exists(args['log_path']+'/logfile.log'):
    with open(args['log_path']+'/logfile.log','r') as f:
      fstr = f.read()
    numlines = len(fstr.split('\n'))
  else:
    numlines=None
  results['numloglines']=numlines
  print('NUUUUUUUUM LINES', numlines)
  ############
  
  ###########################
  ####!!!!!!!!!!!!!!!!!!!!
  args['coherent']=1.0
  args['sparsity']=0.0
  args['costover']=0.0
  ####################
  
  ##############################################
  tottest=0;totdev=0;
  eb =0
  if 1:

      ############################################################
      usettrack=Counter()
      if len(args['dev_file'])>0:
        bsizes=[]
        for (by,bx) in data_dev.batch(50).take(1000000000000):
          if 1:
            bsize=np.shape(bx)[0]
            totdev+=bsize

            dev_ddict=just_predict_wdat(jpraw=jpraw,args=args,
                                          model=amodel,
                                          x=bx,
                                          y=by,
                                          train=False,
                                          bsize=bsize)        
                                        
            bx = bx.numpy()
            bsizes.append(bsize)
            devdict = update_ldict(devdict,dev_ddict)
            usettrack = zwcounts(dev_ddict,bx,usettrack)

            
      results['used_track']=dict(usettrack)
      
      for k in devdict:
          if 'z' not in k and 'pred' not in k:            
            results['dev'][k] = np.dot(bsizes,devdict[k])/np.sum(bsizes)
            print('DEV', k, results['dev'][k])
      #################################################################
       
             
      if len(args['finedev_file'])>0:
        finebsizes=[]
        for (by,bx) in data_finedev.batch(50).take(1000000000000):
          if 1:
            bsize=np.shape(bx)[0]
            totdev+=bsize

            dev_ddict=just_predict_wdat(jpraw=jpraw,args=args,
                                          model=amodel,
                                          x=bx,
                                          y=by,
                                          train=False,
                                          bsize=bsize)        
                                        
            bx = bx.numpy()
            finebsizes.append(bsize)
            finedevdict = update_ldict(finedevdict,dev_ddict)
            
        for k in finedevdict:
            if 'z' not in k and 'pred' not in k:            
              results['finedev'][k] = np.dot(finebsizes,finedevdict[k])/np.sum(finebsizes)
              print('fineDEV', k, results['finedev'][k])
       
      ##############################       

  
      ## 
      if 1:
        ## evaluation
        ei=0
        bsizes=[];
        prec1s=[[] for i in range(len(args['aspects']))]
        prec2s=[[] for i in range(len(args['aspects']))]
        zlens=[[] for i in range(len(args['aspects']))]
        bytrack=[]
        for by,bx,tz in dataset.batch(1).take(500000000000):
            byn=by.numpy()[0];tzn=tz.numpy()[0]
            if (byn>=.6 or byn<=.4) and sum(tzn)>0:
              bytrack.append(byn[0])

              tz = tz.numpy()
              
              
              bsize=np.shape(bx)[0]
              tottest+=bsize

              dev_ddict=just_predict_wdat(jpraw=jpraw,args=args,
                                            model=amodel,
                                            x=bx,
                                            y=by,
                                            train=tf.constant(False,dtype=tf.bool),
                                            bsize=bsize)        
              dev_ddict['x']=bx.numpy()
              dev_ddict['tz']=tz
              bsizes.append(bsize)
              

              tdict = update_ldict(tdict,dev_ddict)
              
              if len(args['savename'])>0:
                by = by.numpy()
                
                with open(args['savename'],'a') as f:
                  for i in range(bsize):             
              
                    x1 = embed_layer.map_to_words(bx[i,:])
                    xstr = ' '.join([x for x in x1 
                                  if x!='<padding>' and x!= '<start>' and x!='<end>'])
              
                    zstr='';predstr='';
                    tzstr ='';ystr='';
                    for ai in range(len(args['aspects'])):
                      zstr += '\t'+' '.join([str(zi) for zi in dev_ddict['z'+str(ai)][i,:]]) 
                      if args['binarize']:
                        predstr +='\t'+ str(dev_ddict['preds'+str(ai)][i,1])
                      else:
                        predstr +='\t'+ str(dev_ddict['preds'+str(ai)][i])                      

                      tzstr += '\t'+' '.join([str(zi) for zi in tz[i,:,ai]]) 
                      ystr +='\t'+ str(by[i,ai])

                    dumpstr = '\t'.join([ystr,predstr,xstr,tzstr,zstr])+'\n'
                    f.write(dumpstr)            

        if 1:
          for a in range(len(args['aspects'])):
            (prec1,prec2,
            recall1,recall2,
            f1s1,f1s2,
              zlen) = get_prec(np.vstack(tdict['x']),np.vstack(tdict['z'+str(a)]),
                                  np.vstack(tdict['tz'])[:,:,a],
                                    padid=amodel.generator.padding_id)
            results['test']['prec1_'+str(a)] = prec1
            results['test']['prec2_'+str(a)] = prec2
            results['test']['recall1_'+str(a)] = recall1
            results['test']['recall2_'+str(a)] = recall2            
            results['test']['f1s1_'+str(a)] = f1s1
            results['test']['f1s2_'+str(a)] = f1s2            
            print('PRECS', prec1,prec2)
            print('recall', recall1,recall2)
            print('f1', f1s1,f1s2)

        byt2=[]
        for yi in bytrack:
          if yi>=.6:
            byt2.append(1)
          
          else:
            byt2.append(0)
        print(Counter(bytrack))
        countsby=Counter(byt2)
        results['test']['ycounts'] = (countsby[0],countsby[1])
        print(results['test']['ycounts'])
        for k in tdict:          
          if 'z' not in k and 'pred' not in k and k!='x':
            results['test'][k] = np.dot(bsizes,tdict[k])/np.sum(bsizes)

  print('TOTDEV', totdev, 'TOTEST', tottest)

  with open(args['res_file']+'.aug','a') as f: #!!!!!!!!!
    astr = json.dumps(results)
    f.write(astr)    
    f.write('\n')




