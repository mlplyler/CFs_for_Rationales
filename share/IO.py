####################################################
## IO.py
####################################################

# -*- coding: utf-8 -*-
"""
most of this is inherited from other repos...
"""

import numpy as np
import sys
import gzip
import random
import json
import os

import tensorflow as tf


from collections import Counter,defaultdict
####################################

####################################
####################################

## keep!
def make_a_jsongen(fname,aspects,embed_layer,
                maxlen=256,addstartend=0,binit=0,splitnum=.5):
  def jsongen():
        with open(fname) as f:
          for line in f:
            thej = json.loads(line)

            if 1:#binit:
              scores=[]

              for aspect in aspects:
                score1 = float(thej['y'][aspect])
                if binit:
                  if score1>=splitnum:
                    score1=1.0
                  else: 
                    score1=0.0

                scores.append(score1) 
            text_list = thej['x']
            id_list = fix_maxlen(text_list,
                                      embed_layer,
                                      maxlen,
                                        addstartend=addstartend)
            zs=[]
            for aspect in aspects:
              ainds = thej[str(aspect)] 
              z = [0.0 for i in range(maxlen)]
              for rtup in ainds:
                for i in range(maxlen):
                  if i>=rtup[0] and i<rtup[1]:
                      z[i]=1.0
              zs.append(z)
            zs = np.array(zs).T                
            yield(scores,id_list,zs)
  dataset = tf.data.Dataset.from_generator(
     jsongen,     
     ((tf.float32, tf.int32, tf.float32)),
     (tf.TensorShape([len(aspects)]), tf.TensorShape([None]), (tf.TensorShape([maxlen,len(aspects)]))),
     )
  return(dataset)  
  
####################################
####################################
## keep!
def make_a_hotelgen(fname,aspects,embed_layer,maxlen=256,addstartend=0,binit=0,splitnum=.5):
  def jsongen():
        with open(fname) as f:
          for li,line in enumerate(f):
            if li>0: ## skip the header line!!                          
              splitline=line.split('\t')              
              if 1:
                scores=[]
                for aspect in aspects:
                  score1 = float(splitline[1])                  
                  if binit:
                    if score1>=splitnum:
                      score1=1.0
                    else: 
                      score1=0.0

                  scores.append(score1) 
              text_list = splitline[2].split()
              id_list = fix_maxlen(text_list,
                                        embed_layer,
                                        maxlen,
                                          addstartend=addstartend)
              zs=[]
              
              z = [int(i) for i in splitline[3].split()]
              for i in range(maxlen-len(z)): ## add padding to z
                z.append(0)
              z = z[:maxlen]## trim if too big    
              zs.append(z)
              zs = np.array(zs).T                              
              yield(scores,id_list,zs)
  dataset = tf.data.Dataset.from_generator(
     jsongen,     
     ((tf.float32, tf.int32, tf.float32)),
     (tf.TensorShape([len(aspects)]), tf.TensorShape([None]), (tf.TensorShape([maxlen,len(aspects)]))),
     )
  return(dataset)  
  
## keep!  
def make_a_maskgen_jrand(fname,aspects,embed_layer,
    maxlen=256,addstartend=0,binit=1,perchigh=.10,classpick=None):  
  def ratdatagen():
        with open(fname) as f:        
          for line in f:
            breakline=False          
            if '\t' in line:
              ## if classpick isnt 1/0 dont worry about this part
              if classpick is not None or (classpick!=0 and classpick!=1):
                if binit:
                  scores=[]
                  for aspect in aspects:
                    score1 = float(line.split('\t')[0].split()[aspect])
                    if classpick!=-2:
                      if score1>=.6:
                        score1=1.0
                      elif score1<=.4:
                        score1=0.0
                      else:
                        breakline=True
                        break
                    else:
                      score1=0.0
                    if classpick is not None or classpick!=-1:
                      if (classpick==1 and score1==0) or (classpick==0 and score1==1):
                        breakline=True
                        break 
                    scores.append(score1) 
                else:
                  scores=[]  
                  for aspect in aspects:
                    score1 = float(line.split('\t')[0].split()[aspect])
                    scores.append(score1) 
              if breakline:
                continue
              text_list = line.split('\t')[1].split()
              if len(text_list)> 1:
                id_list = fix_maxlen(text_list,
                                          embed_layer,
                                          maxlen,
                                            addstartend=addstartend)
                ## mask a masked span
                thetok = '<zero>' ## hardzero only function here bro
                lentext = min([len(text_list),maxlen])  
                numhighs = int(perchigh*lentext)                
                textlistm=list(text_list)                
                mlis = sorted(list(set(random.sample(range(lentext),k=numhighs))))
                maskvec=[]                 
                for li in range(lentext):
                  if li in mlis:
                    maskvec.append(1)
                    randr = random.randint(0,9)
                    if randr==0:
                      textlistm[li]=textlistm[li]
                    elif randr==1:
                      ## hardcode for these embedding <>
                      theint = random.randint(6,embed_layer.n_V) 
                      textlistm[li]=embed_layer.map_to_words(
                                [theint])[0]
                    else:
                      textlistm[li]=thetok
                  else:
                    maskvec.append(0)
                for i in range(len(maskvec),maxlen):
                  maskvec.append(0)
                idlistm = fix_maxlen(textlistm,embed_layer,maxlen,addstartend=addstartend)                  
                yield(idlistm,id_list,maskvec)              
  dataset = tf.data.Dataset.from_generator(
     ratdatagen,     
     (( tf.int32,tf.int32,tf.float32)),
     ( tf.TensorShape([None]),
     tf.TensorShape([None]),tf.TensorShape([None])),
     )
  return(dataset)
## keep!     
def make_a_maskgen_jcont(fname,aspects,embed_layer,
    maxlen=256,addstartend=0,binit=1,perchigh=.10,classpick=None):
  def ratdatagen():
        with open(fname) as f:        
          for line in f:
            breakline=False          
            if '\t' in line:
              ## if classpick isnt 1/0 dont worry about this part
              if classpick is not None or (classpick!=0 and classpick!=1):
                if binit:
                  scores=[]
                  for aspect in aspects:
                    score1 = float(line.split('\t')[0].split()[aspect])
                    if classpick!=-2:
                      if score1>=.6:
                        score1=1.0
                      elif score1<=.4:
                        score1=0.0
                      else:
                        breakline=True
                        break
                    else:
                      score1=0.0
                    if classpick is not None or classpick!=-1:
                      if (classpick==1 and score1==0) or (classpick==0 and score1==1):
                        breakline=True
                        break 
                    scores.append(score1) ############### split on .6 only!!!!!!!!
                else:
                  scores=[]  
                  for aspect in aspects:
                    score1 = float(line.split('\t')[0].split()[aspect])
                    scores.append(score1) ############### split on .6 only!!!!!!!!
              if breakline:
                continue
              text_list = line.split('\t')[1].split()
              if len(text_list)> 1:
                id_list = fix_maxlen(text_list,
                                          embed_layer,
                                          maxlen,
                                            addstartend=addstartend)
                ## mask a masked span
                thetok = '<zero>' ## hardzero only function here bro
                lentext = min([len(text_list),maxlen])  
                numhighs = int(perchigh*lentext)                
                textlistm=list(text_list)                
                start_ind = random.randint(0,lentext-numhighs)                
                mlis = range(start_ind,start_ind+numhighs)                
                maskvec=[]  
                #for li in mlis:
                for li in range(lentext):
                  if li in mlis:
                    maskvec.append(1)
                    randr = random.randint(0,9)
                    if randr==0:
                      textlistm[li]=textlistm[li]## keep the same
                    elif randr==1:
                      ## hardcode for these embedding <>
                      theint = random.randint(6,embed_layer.n_V) 
                      textlistm[li]=embed_layer.map_to_words(
                                [theint])[0]## get a random word....
                    else:
                      textlistm[li]=thetok
                  else:
                    maskvec.append(0)
                for i in range(len(maskvec),maxlen):
                  maskvec.append(0)
                idlistm = fix_maxlen(textlistm,embed_layer,maxlen,addstartend=addstartend)                  
                yield(idlistm,id_list,maskvec)              
  dataset = tf.data.Dataset.from_generator(
     ratdatagen,     
     (( tf.int32,tf.int32,tf.float32)),
     ( tf.TensorShape([None]),
     tf.TensorShape([None]),tf.TensorShape([None])),
     )
  return(dataset)   



## keep!
def make_a_3gen(fname,aspects,embed_layer,maxlen=256,addstartend=0,binit=1):
  def ratdatagen():
        with open(fname) as f:        
          for line in f:
            breakline=False
            if '\t' in line:
              if binit:
                scores=[]
                for aspect in aspects:
                  score1 = float(line.split('\t')[0].split()[aspect])
                  if score1>=.6:
                    score1=1.0
                  elif score1<=.4:
                    score1=0.0
                  else:
                    breakline=True
                    break
                  scores.append(score1) ############### split on .6 only!!!!!!!!
              else:
                scores=[]  
                for aspect in aspects:
                  score1 = float(line.split('\t')[0].split()[aspect])
                  scores.append(score1) ############### split on .6 only!!!!!!!!
              if breakline:
                continue

              text_list = line.split('\t')[1].split()
              if len(text_list)> 1:
                id_list = fix_maxlen(text_list,
                                          embed_layer,
                                          maxlen,
                                            addstartend=addstartend)                
                yield(scores,id_list)
              
  dataset = tf.data.Dataset.from_generator(
     ratdatagen,     
     ((tf.float32, tf.int32)),
     (tf.TensorShape([len(aspects)]), tf.TensorShape([None])),
     )
  return(dataset)

## keep!
def make_a_fcgen_sorted_track(fname,aspects,embed_layer,
      maxlen=256,addstartend=0,binit=1,classpick=None,keepind=None):
  def ratdatagen():        
        with open(fname) as f:   
          lines = list(f.read().split('\n'))
          lens = [len(t.split(' ')) for t in lines]          
          
          for llii in np.argsort(lens)[::-1]: 
            if llii in keepind:
              line = lines[llii] 
              
              if '\t' in line:
                
                if binit:
                  scores=[]
                  
                  for aspect in aspects:
                    score1 = float(line.split('\t')[0].split()[aspect])
                    if score1>=.6:
                      score1=1.0
                    elif score1<=.4:
                      score1=0.0
                    else:
                      
                      continue
                    if classpick is not None or classpick!=-1:
                      if (classpick==1 and score1==0) or (classpick==0 and score1==1):
                        continue

                    scores.append(score1) ############### split on .6 only!!!!!!!!
                    
                else:
                  scores=[]  
                  for aspect in aspects:
                    score1 = float(line.split('\t')[0].split()[aspect])
                    scores.append(score1) ############### split on .6 only!!!!!!!!
                
                text_list = line.split('\t')[1].split()
                if len(text_list)> 1 and len(scores)>0:
                  id_list = fix_maxlen(text_list,
                                            embed_layer,
                                            maxlen,
                                              addstartend=addstartend)
                  
                  yield(scores,id_list,llii)
                  
  keepind=keepind
  dataset = tf.data.Dataset.from_generator(
     ratdatagen,     
     ((tf.float32, tf.int32,tf.int32)),
     (tf.TensorShape([len(aspects)]), tf.TensorShape([None]),()),
     )
  return(dataset)


## keep!
def make_a_fcgen(fname,aspects,embed_layer,
      maxlen=256,addstartend=0,binit=1,classpick=None):
  def ratdatagen():
        with open(fname) as f:             
          for line in f:          
            if '\t' in line:
              if binit:
                scores=[]
                for aspect in aspects:
                  score1 = float(line.split('\t')[0].split()[aspect])
                  if score1>=.6:
                    score1=1.0
                  elif score1<=.4:
                    score1=0.0
                  else:
                    continue
                  if classpick is not None or classpick!=-1:
                    if (classpick==1 and score1==0) or (classpick==0 and score1==1):
                      continue
                      
                  scores.append(score1) ############### split on .6 only!!!!!!!!
                  
              else:
                scores=[]  
                for aspect in aspects:
                  score1 = float(line.split('\t')[0].split()[aspect])
                  scores.append(score1) ############### split on .6 only!!!!!!!!
          
              text_list = line.split('\t')[1].split()
              if len(text_list)> 1 and len(scores)>0:
                id_list = fix_maxlen(text_list,
                                          embed_layer,
                                          maxlen,
                                            addstartend=addstartend)
                
                yield(scores,id_list)
              
  dataset = tf.data.Dataset.from_generator(
     ratdatagen,     
     ((tf.float32, tf.int32)),
     (tf.TensorShape([len(aspects)]), tf.TensorShape([None])),
     )
  return(dataset)
################################
######################
def fix_maxlen(text_list,embed_layer,maxlen,
        addstartend=False):
  if len(text_list)>=(maxlen):
    if addstartend:
      text_list = text_list[:maxlen]
      text_list = ['<start>'] +\
                   text_list[:-2] +\
                  ['<end>']
    else:
      text_list = text_list[:maxlen]
    wl_list = embed_layer.map_to_ids(text_list)
    return(wl_list)
  else:
    text_list = text_list
    if addstartend:
      if len(text_list)>(maxlen-2):
        text_list =  ['<start>'] +\
                   text_list[:-2] +\
                  ['<end>']
      else:
        text_list =  ['<start>'] +\
                   text_list +\
                  ['<end>']

    id_list = embed_layer.map_to_ids(text_list)
    id_list = np.hstack([id_list,
                        np.array([embed_layer.vocab_map['<padding>']
                                  for k in range(maxlen-len(id_list))])])
    return(id_list)
  
def fix_startend(strlist):
  if strlist[0]=='<start>':
    strlist = strlist[1:]
    strlist.append('<padding>')
  if '<end>' in strlist:
    theind = strlist.index('<end>')
    strlist[theind] = '<padding>'
  return(strlist)
###########
###############
def printnsay(thefile,text):
  print(text)
  with open(thefile,'a') as f:
    f.write(text+'\n')
def justsay(thefile,text):
  with open(thefile,'a') as f:
    f.write(text+'\n')
