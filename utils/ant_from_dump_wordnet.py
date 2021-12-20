import json
import numpy as np
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
from multiprocessing import Pool
import nltk
from nltk.corpus import wordnet
import nltk


def file_to_stuff_bare(fname):
  with open(fname,'r') as f:
      fstr = f.read()
  flines = fstr.split('\n')[:-1]
  print('num lines', len(flines))
  sons = []
  for i,fl in enumerate(flines):
      try:
          adict = json.loads(fl)
          if len(adict.keys())==8:
              sons.append(adict)
      except:
          print(i)
      if len(sons[-1].keys())<8:
          print(i, len(sons[-1].keys()),sons[-1].keys())

  print('num', len(sons))
  sons[0].keys()

  ys = np.array([float(s['y']) for s in sons])
  preds = np.array([float(s['pred']) for s in sons])
  pbs = np.array([1 if p>=.5 else 0 for p in preds])
  zs = np.array([[1.0 if float(t)>=.5 else 0.0 for t in s['z']] for s in sons])

  texts = np.array([[t for t in s['x'] 
            if (t!='<padding>' and t!='<start>' and t!='<end>')
                  ] for s in sons])
  tlens = np.array([len(t) for t in texts])

  ## straight up
  cfs = np.array([[t for i,t in enumerate(s['cf'] )
            if (t!='<padding>' and t!='<start>' and t!='<end>')
                  ] for s in sons])
  allstuff = []                  
  for i in range(len(ys)):
    allstuff.append({'y':ys[i],
                     'z':zs[i],
                     'text':texts[i]})

  return(allstuff)


### NEED TO GRAB A NEW ONE IF THE FIRST HAS NO RAT
### handle starting or trailing spaces
def get_ants(aword):
  antonyms = []    
  for syn in wordnet.synsets(aword):
      for l in syn.lemmas():
          #synonyms.append(l.name())
          if l.antonyms():
              antonyms.append(l.antonyms()[0].name().lower())  
  return(list(set(antonyms)))

def rat_ant_flipper(wl,zl,vocab,
                    changepos=['IN','JJ','JJR','JJS']):
    posl = partofspeech_that_POS([wl])[0]
    flipwl=list(wl)
    for i in range(len(wl)):        
        if zl[i] and wl[i] not in stopset:#and posl[i] in changepos:
            ## attempt the flip
            ants = get_ants(wl[i])
            if type(ants) is list:
                for a in ants:
                    a = a.lower()
                    if a in vocab:
                        cwl = list(wl)
                        cwl[i]=a
                        newposl = partofspeech_that_POS([cwl])[0]
                        if posl[i]==newposl[i]:
                            flipwl[i]=a
                            break                     

                            
    return(' '.join(flipwl))
                        
def find_ant_wrap(stuff):
  return(rat_ant_flipper(stuff[0],stuff[1],stuff[2],))

def remove_start_end(alist):
    return([w for w in alist if w!='<start>' and w!='<end>'])

def get_vocab(embfile):
  ## grab your vocab dog
  wlist = []
  with open(embfile,'r') as f:
      for i,line in enumerate(f):
          w=line.split(' ')[0].replace('\n','')
          wlist.append(w)
          if i>=32768-7:
            break
  vocab = set(wlist)        
  return(vocab)


## need to get get part of speech tagger going
def partofspeech_that_POS(alist):
    pos_list = map(lambda x: nltk.pos_tag(x),alist)
    just_pos = map(lambda x: [z[1] for z in x],pos_list)    
    return(list(just_pos))
#################################################################
#################################################################
#################################################################
#################################################################
#################################################################
#############################################
if __name__=='__main__':
  '''
  this script requires that you pass a counterfactual dump file
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument('thefile')  
  parser.add_argument('embfile')
  parser.add_argument('savefile')
  parser.add_argument('aspect')
  targs = parser.parse_args()

  stopset = set(['.',',','!','?',':',';','-',')','(','&',
            '<unk>','if','what','beer','that','has',
            "n't",'this','which','though','should','quite',
            'into','was','we'])
  
  print('getting vocab')
  vocab=get_vocab(targs.embfile)
  print('WORDS', 'appear\n' in vocab, 'black\n' in vocab)
  print('the vocab', targs.embfile, len(vocab))
  print('loading stuff...')
  ## this is the stuff you want to flip
  thestuff = file_to_stuff_bare(targs.thefile)
  print('UNLIMITED POWER OF {} cores'.format(multiprocessing.cpu_count()))
  apool = Pool(multiprocessing.cpu_count())
  fmw_list = [(thestuff[i]['text'],thestuff[i]['z'],vocab)
                       for i in range(len(thestuff))]
  result_strings = apool.map(find_ant_wrap,fmw_list)
  print('lets save this to a file man')
        
  dlist=[]
  for i in range(len(thestuff)):
    if targs.aspect =='0':
      dlist.append(str(thestuff[i]['y'])+'\t'+result_strings[i]) ## careful now!!!!
      dlist.append(str(1-thestuff[i]['y'])+'\t'+' '.join(thestuff[i]['text'])) ## careful now!!!!
    elif targs.aspect =='1':
      dlist.append('69 '+str(thestuff[i]['y'])+'\t'+result_strings[i]) ## careful now!!!!
      dlist.append('59 '+str(1-thestuff[i]['y'])+'\t'+' '.join(thestuff[i]['text'])) ## careful now!!!!
    elif targs.aspect == '2':
      dlist.append('69 69 '+str(thestuff[i]['y'])+'\t'+result_strings[i]) ## careful now!!!!
      dlist.append('69 69 '+str(1-thestuff[i]['y'])+'\t'+' '.join(thestuff[i]['text'])) ## careful now!!!!      

    
  dumpstr = '\n'.join(dlist)
  with open(targs.savefile,'w') as f:
    f.write(dumpstr)
  


















