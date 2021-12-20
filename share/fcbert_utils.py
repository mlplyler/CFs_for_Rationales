
import tensorflow as tf
import numpy as np
import json

from embeddings_novector import *
sys.path.append('../share/')
from IO import printnsay

def load_jrat(args,Jrat_Model=None,embed_layer=None,amodel=None,chkptdir=None):

  if len(args['load_chkpt_dir'])>0:
    chkptdir=args['load_chkpt_dir']
    with open(args['load_chkpt_dir']+'/config.json','r') as f:
          cstr = f.read()
    args2 = json.loads(cstr)        
    
    for k in ['tnlayers','hidden_dimension','tdff','theads']:#,'n_v']:
      if k in args2:
        args[k]=args2[k]
      else:
        print('load jrat missing arg', k)
    for k in args:  
      if args[k] is None and k in args2 and k!='n_v':
        args[k]=args2[k]

  ## load embeddings
  if embed_layer is None:    
    embed_layer = create_embedding_layer(args['embedding'],n_d=args['embdim'],v_max=args['n_v'])
  ## init a model
  if amodel is None:
    amodel = Jrat_Model(args,embed_layer)
  ## setup chkpt manager
  theckpt = tf.train.Checkpoint(step=tf.Variable(1),
                              net=amodel)
  chkptman = tf.train.CheckpointManager(theckpt,
                                        args['log_path'],
                                        max_to_keep=1)    
  ## load model if provided                                        
  if chkptdir is not None:
    print('TRYING TO LOAD', chkptdir)
    theckpt.restore(
                tf.train.latest_checkpoint(chkptdir)
                            ).assert_nontrivial_match()
    print('LOADED MODEL', chkptdir)    
  ## set up optimizers
  if args['gen_lr'] == -1:
    glr =  CustomSchedule(args['hidden_dimension'])
  else:  
    glr = tf.Variable(args['gen_lr'],dtype=tf.float32)
  if args['enc_lr'] ==  -1:
    elr =  CustomSchedule(args['hidden_dimension'])  
  else:
    elr = tf.Variable(args['enc_lr'],dtype=tf.float32)
  
  ogen = tf.keras.optimizers.Adam(learning_rate=glr) 
  oenc = tf.keras.optimizers.Adam(learning_rate=elr)

  jratd = {'amodel':amodel,
          'theckpt':theckpt,
          'chkptman':chkptman,
          'opterg':ogen,
          'optere': oenc,
          'elayer':embed_layer}
  return(jratd,args)
  
  
  
def load_cf(args,CF_Model=None,embed_layer=None,amodel=None,chkptdir=None,
            cftext = 'cfmodel',dub_opt=False,logfile=None):

  if chkptdir is None and len(args['load_chkpt_dir'])>0 and ('cf_chkpt_dir' not in args or 
                   len(args['cf_chkpt_dir'])==0) and ('cfboth_chkpt_dir' in args and len(args['cfboth_chkpt_dir'])>0):
      chkptdir=args['cfboth_chkpt_dir']+'/'+cftext+'/'
      print('cfboth     CHKPTDIR222222', chkptdir)          
      
      with open(args['cfboth_chkpt_dir']+'/config.json','r') as f:
          cstr = f.read()
      args_cf = json.loads(cstr)        
      
      for k in args_cf:
        if 'cf_' in k and k!='cf_lr':
          args[k]=args_cf[k]

  elif ('cf_chkpt_dir'  in args and
     len(args['cf_chkpt_dir'])>0):
          
      with open(args['cf_chkpt_dir']+'/../config.json','r') as f:
          cstr = f.read()
      args_cf = json.loads(cstr)        
      
      for k in args_cf:
        if 'cf_' in k and k!='cf_chkpt_dir':
          args[k]=args_cf[k]

      chkptdir=args['cf_chkpt_dir']
      print('CHKPTDIR33333333', chkptdir)      
  elif chkptdir is not None:
      with open(chkptdir+'/../config.json','r') as f:
          cstr = f.read()
      args_cf = json.loads(cstr)        
      
      for k in args_cf:
        if 'cf_' in k:
          print('refresh', k, args_cf[k])
          args[k]=args_cf[k]
      print('refreshed the arggggggs')    
  else:
    print('cf no load conditions',      
      args['cfboth_chkpt_dir'],
      args['load_chkpt_dir'])
      
  ## load embeddings
  if embed_layer is None:    
    embed_layer = create_embedding_layer(args['embedding'],
                  n_d=args['embdim'],v_max=args['n_v'])    
    if args['n_v']==None:
      args['n_v'] = len(embed_layer.vocab_map)
  print('CF LOAD N_V',len(embed_layer.vocab_map))
  print()    
  ## init a model
  if amodel is None:
    amodel = CF_Model(args,embed_layer)
  ## setup chkpt manager
  theckpt = tf.train.Checkpoint(step=tf.Variable(1),
                              net=amodel)
  if not os.path.exists(args['log_path']+'/'+cftext+'/'):
    try:
      os.makedirs(args['log_path']+'/'+cftext+'/')
    except:
      print('already got!!')
  if 'chkps_tokeep' in args:
    to_keep=args['chkps_tokeep']    
  else:
    to_keep=1
  chkptman = tf.train.CheckpointManager(theckpt,
                                        args['log_path']+'/'+cftext+'/',
                                        max_to_keep=to_keep)    
  ## load model if provided                                              
  if chkptdir is not None and os.path.exists(chkptdir): 
    print('chkptdir', chkptdir)
    theckpt.restore(
                tf.train.latest_checkpoint(chkptdir)
                            ).assert_nontrivial_match()
    print('LOADED MODEL', chkptdir)
    if logfile is not None:
      printnsay(thefile=logfile,text='LOADED MODEL'+chkptdir)
  else:
    print('cf no load', chkptdir)
    if logfile is not None:
      printnsay(thefile=logfile,text='cf no load' + str(chkptdir))

  ## set up optimizers  
  if args['cf_lr'] == -1:
    cflr =  CustomSchedule(args['cf_dmodel'])
  elif args['cf_lr']==-2:
    cflr = CustomScheduleBert(warmup_steps=args['warmup'],peak_lr=args['peak_lr'])
  else:  
    cflr = tf.Variable(args['cf_lr'],dtype=tf.float32)

  ocf = tf.keras.optimizers.Adam(learning_rate=cflr)
  if dub_opt:
    ocf2 = tf.keras.optimizers.Adam(learning_rate=cflr) 
    cfd = {'amodel':amodel,
          'theckpt':theckpt,
          'chkptman':chkptman,
          'opter':ocf,
          'opter2':ocf2,
          'elayer':embed_layer}
  
  else:
    cfd = {'amodel':amodel,
          'theckpt':theckpt,
          'chkptman':chkptman,
          'opter':ocf,
          'elayer':embed_layer}
  return(cfd,args)
  
def load_cf2(args,CF_Model=None,embed_layer=None,amodel=None,chkptdir=None,cftext = 'cfmodel'):
  with open(chkptdir+'/../config.json','r') as f:
      cstr = f.read()
  args_cf = json.loads(cstr)        

  ## load embeddings
  if embed_layer is None:
    #embed_layer = LOAD_EMBED_LAYER()
    embed_layer = create_embedding_layer(args_cf['embedding'],
        n_d=args_cf['embdim'],v_max=args_cf['n_v'])    
  ## init a model
  if amodel is None:
    amodel = CF_Model(args_cf,embed_layer)
  ## setup chkpt manager
  theckpt = tf.train.Checkpoint(step=tf.Variable(1),
                              net=amodel)
  if not os.path.exists(args_cf['log_path']+'/'+cftext+'/'):
    os.makedirs(args_cf['log_path']+'/'+cftext+'/')
  chkptman = tf.train.CheckpointManager(theckpt,
                                        args_cf['log_path']+'/'+cftext+'/',
                                        max_to_keep=1)    
  ## load model if provided                                              
  if chkptdir is not None and os.path.exists(chkptdir): 
    print('chkptdir', chkptdir)
    theckpt.restore(
                tf.train.latest_checkpoint(chkptdir)
                            ).assert_nontrivial_match()
    print('LOADED MODEL', chkptdir)
  else:
    print('cf no load', chkptdir)
  ## set up optimizers
  if args_cf['cf_lr'] == -1:
    cflr =  CustomSchedule(args_cf['cf_dmodel'])
  else:  
    cflr = tf.Variable(args_cf['cf_lr'],dtype=tf.float32)

  ocf = tf.keras.optimizers.Adam(learning_rate=cflr)
  
  cfd = {'amodel':amodel,
          'theckpt':theckpt,
          'chkptman':chkptman,
          'opter':ocf,
          'elayer':embed_layer}
  return(cfd,args_cf)
  
  
    
  
  
 
  
def cag_wrap_double(cag,args,jratd,cfd0,cfd1,x,y,train=True):
  bsize=np.shape(x)[0]
 
  (costg,coste,loss,obj,pkept,z,
         cost_d,dec_ids,
         sparse,coherent,flex,
         costg_cf,coste_cf,loss_cf,obj_cf,pkept_cf,z_cf,
         coherent_cf,sparse_cf,flex_cf) =cag(args,
                                          jratd['amodel'],cfd0['amodel'],
                                          cfd1['amodel'],x,y,bsize,
                                          jratd['optere'],jratd['opterg'],
                                          cfd0['opter'],cfd1['opter'],train=train) 
  ddict={
        'costg':costg[0].numpy(),
        'coste':coste[0].numpy(),
        'loss':loss[0].numpy(),
        'obj':obj[0].numpy(),
        'pkept':pkept[0].numpy(),
        'costd':cost_d.numpy(),
        'coherent':coherent[0].numpy(),
        'sparse':sparse[0].numpy(),
        'flex':flex[0].numpy(),
        'costg_cf':costg_cf[0].numpy(),
        'coste_cf':coste_cf[0].numpy(),
        'loss_cf':loss_cf[0].numpy(),
        'obj_cf':obj_cf[0].numpy(),
        'pkept_cf':pkept_cf[0].numpy(),
        'coherent_cf':coherent_cf[0].numpy(),
        'sparse_cf':sparse_cf[0].numpy(),
        'flex_cf':flex_cf[0].numpy(),        
        }
          
  return(ddict)    


def cag_wrap_doubleRL(cag,args,jratd,cfd0,cfd1,x,y,train=True):
  bsize=np.shape(x)[0]
 
  (cfpredloss0,cfpredloss1,costd0,costd1) =cag(args,
                                          jratd['amodel'],cfd0['amodel'],
                                          cfd1['amodel'],x,y,bsize,
                                          jratd['optere'],jratd['opterg'],
                                          cfd0['opter'],cfd1['opter'],
                                          cfd0['opter2'],cfd1['opter2'],
                                          train=train) 
  ddict={

        'loss_cf0':cfpredloss0.numpy(),
        'loss_cf1':cfpredloss1.numpy(),
        'costd0':costd0.numpy(),
        'costd1':costd1.numpy(),        
     
        }
  return(ddict) 

def jpredictby1_wrap(thefn,args,jratd,cfd,x,y,train=True,
                          nounk=False,unkid=0,randsamp=0):
  bsize=np.shape(x)[0]  
  (costd,dec_ids,
      newx,newz,newy,newpreds) =thefn(args,
                              jratd['amodel'],cfd['amodel'],
                              x,y,bsize,
                              train=train,
                              nounk=nounk,unkid=unkid,doretrace=args['slevel'],
                              randsamp=randsamp)     
  ddict={
        'costd':costd.numpy(),
        'newx':newx.numpy(),
        'newy':newy.numpy(),
        'newpred':newpreds.numpy(),        
        'newz':newz.numpy(),
        'dec_ids':dec_ids.numpy(),

        }
  return(ddict)  
  
def jpredicts2s_wrap_double(thefn,args,jratd,cfd0,cfd1,x,y,train=True,
                          nounk=False,unkid=0,randsamp=0):
  bsize=np.shape(x)[0]
  (costd,dec_ids,
      newx,newz,newy,newpreds) =thefn(args,
                              jratd['amodel'],cfd0['amodel'],
                              cfd1['amodel'],x,y,bsize,
                              jratd['optere'],jratd['opterg'],
                              cfd0['opter'],cfd1['opter'],train=train,
                              nounk=nounk,unkid=unkid,doretrace=args['slevel'])     
  ddict={
        'costd':costd.numpy(),
        'newx':newx.numpy(),
        'newy':newy.numpy(),
        'newpred':newpreds.numpy(),        
        'newz':newz.numpy(),
        'dec_ids':dec_ids.numpy(),        
        }
  return(ddict)   
  
def js2scheck_wrap_double(thefn,args,jratd,cfd0,cfd1,x,y,train=True):
  bsize=np.shape(x)[0]
 
  (newx,newz,newy,newpred,
  dec_ids,zcf,predcf) =thefn(args,
                              jratd['amodel'],cfd0['amodel'],
                              cfd1['amodel'],x,y,bsize,
                              jratd['optere'],jratd['opterg'],
                              cfd0['opter'],cfd1['opter'],train=train) 
  ddict={
        'newx':newx.numpy(),
        'newy':newy.numpy(),
        'newpred':newpred.numpy(),        
        'newz':newz.numpy(),
        'dec_ids':dec_ids.numpy(),
        'zcf':zcf.numpy(),
        'predcf':predcf.numpy()
        }
  return(ddict)   
  

def jrat_wrap_double(just_predict,args,jratd,x,y,train=False):  
  '''
  just_predict = rat_jdat
  '''
  (costg,coste,zsum,zdiff,pkept,costover,obj,loss,
  sparse,coherent,flex,z)=just_predict(args,jratd['amodel'],x,y,
                              train=train) 
  ddict={
        'costg':costg[0].numpy(),
        'coste':coste[0].numpy(),
        'loss':loss[0].numpy(),
        'obj':obj[0].numpy(),
        'pkept':pkept[0].numpy(),
        'coherent':coherent[0].numpy(),
        'sparse':sparse[0].numpy(),
        'flex':flex[0].numpy(),        
        }
  return(ddict)     


  
def jp_wrap_double(just_predict,args,jratd,cfd0,cfd1,x,y,train=True):
  bsize=np.shape(x)[0]
 
  (costg,coste,loss,obj,pkept,
         cost_d,dec_ids,
         costg_cf,coste_cf,loss_cf,obj_cf,pkept_cf,
         preds,zs,x_cf,zs_cf,preds_cf,
         x_cfmasked)=just_predict(args,
                                          jratd['amodel'],cfd0['amodel'],cfd1['amodel'],x,y,bsize,
                                          train=train) 
  ddict={
        'costg':costg[0].numpy(),
        'coste':coste[0].numpy(),
        'loss':loss[0].numpy(),
        'obj':obj[0].numpy(),
        'pkept':pkept[0].numpy(),
        'costd':cost_d.numpy(),
        'costg_cf':costg_cf[0].numpy(),
        'coste_cf':coste_cf[0].numpy(),
        'loss_cf':loss_cf[0].numpy(),
        'obj_cf':obj_cf[0].numpy(),
        'pkept_cf':pkept_cf[0].numpy(),
        }
  return(ddict)     

def jpredict_wrap_jrat(just_predict,args,jratd,x,y,train=False):
  bsize=np.shape(x)[0]
  (allpreds,z)=just_predict(args,
                            jratd['amodel'],x,bsize=bsize,
                            start_id =jratd['elayer'].vocab_map["<start>"], 
                            end_id = jratd['elayer'].vocab_map["<end>"],
                            pad_id = jratd['elayer'].vocab_map["<padding>"],                            
                            train=train) 
  ddict={
        'x':x.numpy(),
        'y':y[:,0].numpy(),
        'pred':allpreds[0].numpy(),        
        'z':z[0].numpy(),
        }
  return(ddict)     



def pdat_wrap_double(just_predict,args,jratd,cfd0,cfd1,x,y,train=True):
  bsize=np.shape(x)[0]
 
  (costg,coste,loss,obj,pkept,
         cost_d,dec_ids,
         costg_cf,coste_cf,loss_cf,obj_cf,pkept_cf,
         preds,zs,x_cf,zs_cf,preds_cf,
         x_cfmasked)=just_predict(args,
                                          jratd['amodel'],cfd0['amodel'],cfd1['amodel'],x,y,bsize,
                                          train=train) 
                                          
  print('x', np.shape(x), 'xcf', np.shape(x_cf),'zs', np.shape(zs), 'zs_cf', np.shape(zs_cf))                                          
  print('y', np.shape(y))
  ddict={
        'costg':costg[0].numpy(),
        'coste':coste[0].numpy(),
        'loss':loss[0].numpy(),
        'obj':obj[0].numpy(),
        'pkept':pkept[0].numpy(),
        'costd':cost_d.numpy(),
        'costg_cf':costg_cf[0].numpy(),
        'coste_cf':coste_cf[0].numpy(),
        'loss_cf':loss_cf[0].numpy(),
        'obj_cf':obj_cf[0].numpy(),
        'pkept_cf':pkept_cf[0].numpy(),
        'preds':preds[0].numpy(),
        'zs':zs[0].numpy(),
        'preds_cf':preds_cf[0].numpy(),
        'zs_cf':zs_cf[0].numpy(),
        'x_cf':x_cf.numpy(),       
        'x':x.numpy(),
        'y':y[:,0].numpy(),
        'x_cfmasked':x_cfmasked.numpy()
         
        }
  return(ddict)   
  
                       
def jp_raw(args,jratd,cfd,x,y):  
  return(0)                        
                        
                        
                        
                        
                        
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
                        
                        
class CustomScheduleBert(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, warmup_steps=10000,down_steps=200000,peak_lr=1e-4):
    super(CustomScheduleBert, self).__init__()

    self.start_lr = 1e-12
    
    self.warmup_steps = warmup_steps
    self.down_steps = down_steps
    self.peak_lr = peak_lr
    
    self.wstep = (peak_lr-self.start_lr)/warmup_steps
    self.dstep = (peak_lr-self.start_lr)/(down_steps-warmup_steps)
    self.tracker=0

  @tf.function
  def __call__(self, step):
    if step<=self.warmup_steps:
      lr = self.start_lr+self.wstep*step
    else:
      lr = self.peak_lr-self.dstep*(step-self.warmup_steps)
    return(lr)                     
                 
                        
                        
                        
                        
  
def cag_wrap(cag,args,jratd,cfd,x,y,train=True):
  bsize=np.shape(x)[0]
 
  (costg,coste,loss,obj,pkept,z,
    cost_d,dec_ids,
    costg_cf,coste_cf,loss_cf,obj_cf,pkept_cf,z_cf)=cag(args,
                                          jratd['amodel'],cfd['amodel'],x,y,bsize,
                                          jratd['optere'],jratd['opterg'],cfd['opter'],train=train) 
  ddict={
        'costg':costg[0].numpy(),
        'coste':coste[0].numpy(),
        'loss':loss[0].numpy(),
        'obj':obj[0].numpy(),
        'pkept':pkept[0].numpy(),
        'costd':cost_d[0].numpy(), 
        'costg_cf':costg_cf[0].numpy(),
        'coste_cf':coste_cf[0].numpy(),
        'loss_cf':loss_cf[0].numpy(),
        'obj_cf':obj_cf[0].numpy(),
        'pkept_cf':pkept_cf[0].numpy(),
        }
        
     
  return(ddict)                           
                        
                        
                        
   
def jp_wrap(just_predict,args,jratd,cfd,x,y,train=True):
  bsize=np.shape(x)[0]
 
  (costg,coste,loss,obj,pkept,
         cost_d,dec_ids,
         costg_cf,coste_cf,loss_cf,obj_cf,pkept_cf,
         preds,zs,x_cf,zs_cf,preds_cf,
         x_cfmasked)=just_predict(args,
                                          jratd['amodel'],cfd['amodel'],x,y,bsize,
                                          train=train) 
  ddict={
        'costg':costg[0].numpy(),
        'coste':coste[0].numpy(),
        'loss':loss[0].numpy(),
        'obj':obj[0].numpy(),
        'pkept':pkept[0].numpy(),
        'costd':cost_d[0].numpy(),
        'costg_cf':costg_cf[0].numpy(),
        'coste_cf':coste_cf[0].numpy(),
        'loss_cf':loss_cf[0].numpy(),
        'obj_cf':obj_cf[0].numpy(),
        'pkept_cf':pkept_cf[0].numpy(),
        }
  return(ddict)   
                         
                        
                        
                        
                        
def pdat_wrap(just_predict,args,jratd,cfd,x,y,train=True):
  bsize=np.shape(x)[0]
 
  (costg,coste,loss,obj,pkept,
         cost_d,dec_ids,
         costg_cf,coste_cf,loss_cf,obj_cf,pkept_cf,
         preds,zs,x_cf,zs_cf,preds_cf,
         x_cfmasked                           )=just_predict(args,
                                          jratd['amodel'],cfd['amodel'],x,y,bsize,
                                          train=train) 
                                          
  print('x', np.shape(x), 'xcf', np.shape(x_cf),'zs', np.shape(zs), 'zs_cf', np.shape(zs_cf))                                          
  print('y', np.shape(y))
  ddict={
        'costg':costg[0].numpy(),
        'coste':coste[0].numpy(),
        'loss':loss[0].numpy(),
        'obj':obj[0].numpy(),
        'pkept':pkept[0].numpy(),
        'costd':cost_d[0].numpy(),
        'costg_cf':costg_cf[0].numpy(),
        'coste_cf':coste_cf[0].numpy(),
        'loss_cf':loss_cf[0].numpy(),
        'obj_cf':obj_cf[0].numpy(),
        'pkept_cf':pkept_cf[0].numpy(),
        'preds':preds[0].numpy(),
        'zs':zs[0].numpy(),
        'preds_cf':preds_cf[0].numpy(),
        'zs_cf':zs_cf[0].numpy(),
        'x_cf':x_cf.numpy(),       
        'x':x.numpy(),
        'y':y[:,0].numpy(),
        'x_cfmasked':x_cfmasked.numpy()
         
        }
  return(ddict)   


def cf_single_wrap(thefn,args,cfd,x,y,train=True):
  bsize=np.shape(x)[0]
 
  dec_ids,costd = thefn(args,cfd['amodel'],x,y,bsize,cfd['opter'],train)
                                          
  ddict={
        'costd':costd.numpy(),
        }
  return(ddict)    
  
def cf_single_wrap_decids(thefn,args,cfd,x,y,train=True):
  bsize=np.shape(x)[0]
 
  dec_ids,costd = thefn(args,cfd['amodel'],x,y,bsize,cfd['opter'],train)
                                          
  ddict={
        'costd':costd,
        'dec_ids':dec_ids,
        }
  return(ddict)    
  
    
                        
                        
