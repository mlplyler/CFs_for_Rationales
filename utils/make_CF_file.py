import os
import json
import numpy as np
from collections import Counter
import scipy.stats
import argparse

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('thefile')  
  parser.add_argument('aspect')
  parser.add_argument('-flipit',default='1')

  targs = parser.parse_args()

  if targs.flipit=='1' or targs.flipit.lower=='true':
    print('FLIPPINGIT',targs.flipit)
    flipit=True    
  else:
    print('NOTTTT flippign it', targs.flipit)
    flipit=False


  fname = targs.thefile


  with open(fname,'r') as f:
      fstr = f.read()
  flines = fstr.split('\n')[:-1]
  print('num lines', len(flines))
  sons = []
  for i,fl in enumerate(flines):
      try:
          adict = json.loads(fl)
          if len(adict.keys())>=7:
              sons.append(adict)
      except:
          print(i)
      if len(sons[-1].keys())<7:
          print(i, len(sons[-1].keys()),sons[-1].keys())

  print('num', len(sons))
  sons[0].keys()



  ys = np.array([float(s['y']) for s in sons])
  preds = np.array([float(s['pred']) for s in sons])
  pbs = np.array([1 if p>=.5 else 0 for p in preds])
  zs = np.array([[1.0 if float(t)>=.5 else 0.0 for t in s['z']] for s in sons])

  ## dont have to worry about padding, start, end but its fine
  texts = np.array([[t for t in s['x'] 
            if (t!='<padding>' and t!='<start>' and t!='<end>')
                  ] for s in sons])
  tlens = np.array([len(t) for t in texts])

  ## straight up
  cfs = np.array([[t for i,t in enumerate(s['cf'] )
            if (t!='<padding>' and t!='<start>' and t!='<end>')
                  ] for s in sons])

  
  cfpreds = np.array([float(s['pred_cf']) for s in sons])
  pbs_cf = np.array([1 if p>=.5 else 0 for p in cfpreds])
  zs_cf = np.array([[1.0 if float(t)>=.5 else 0.0 for t in s['z_cf']] for s in sons])


  inds0 = np.array(np.where(ys==0)[0])
  inds1 = np.array(np.where(ys==1)[0])
  print('zeros', len(inds0), 'ones', len(inds1))

  indcorrect = np.array(np.where(ys==pbs)[0])
  indwrong = np.array(np.where(ys!=pbs)[0])
  print('accuracy', len(indcorrect)/len(preds))

  indcorrectcf = np.array(np.where(ys==pbs_cf)[0])
  indwrongcf = np.array(np.where(ys==pbs_cf)[0])
  print('accuracy_cf', len(indcorrectcf)/len(pbs_cf))

  zmse =np.mean((zs-zs_cf)**2,axis=1)
  ## correct 1 or 0
  acc1 = np.sum(np.logical_and(ys==1,pbs_cf==1))/len(inds1)
  acc0 = np.sum(np.logical_and(ys==0,pbs_cf==0))/len(inds0)


  meanpkept0 = np.mean([sum(zs[i])/len(texts[i]) 
                  for i in inds0])
  meanpkept1 = np.mean([sum(zs[i])/len(texts[i]) 
                  for i in inds1])
  print('meankept0', meanpkept0, 'meankept1', meanpkept1)
  ## all
  istr = []
  for i in range(len(sons)):
      if targs.aspect=='0':
        ## aspect 0
        if flipit:
          istr.append(str(1-ys[i])+'\t'+' '.join(texts[i]))### FLIP?!?!
        else:
          istr.append(str(ys[i])+'\t'+' '.join(texts[i]))### FLIP?!?!
        istr.append(str(ys[i])+'\t'+' '.join(cfs[i]))
      elif targs.aspect=='1':
        ## aspect 1
        if flipit:
          istr.append('-69 '+str(1-ys[i])+'\t'+' '.join(texts[i]))
        else:
          istr.append('-69 '+str(ys[i])+'\t'+' '.join(texts[i]))
        istr.append('-69 '+str(ys[i])+'\t'+' '.join(cfs[i]))    
      elif targs.aspect=='2': ## aspect 2
        if flipit:
          istr.append('-69 -69 '+str(1-ys[i])+'\t'+' '.join(texts[i]))    
        else: 
          istr.append('-69 -69 '+str(ys[i])+'\t'+' '.join(texts[i]))    
        istr.append('-69 -69 '+str(ys[i])+'\t'+' '.join(cfs[i]))    
  print('len', len(istr)/2,len(preds))
  dstr = '\n'.join(istr)

  with open(fname+'.ratform','w') as f:
      f.write(dstr)
      print('DUMPED')


  repcount0 = Counter()
  repcount1 = Counter()
  for i in range(len(cfs)):
    arep = [t for t,z in zip(cfs[i],zs[i]) if z>=.5]
    if ys[i]==0:
      repcount0.update(arep)
    else:
      repcount1.update(arep)

  from scipy.stats import entropy    

  p0 = np.array(list(repcount0.values()))
  p0 = p0/np.sum(p0)
  print()
  p1 = np.array(list(repcount1.values()))
  p1 = p1/np.sum(p1)

  with open('./dump_results.txt','a') as f:
    f.write(str(acc0)+'__'+str(acc1)+
    '__'+str(entropy(p0))+'__'+str(entropy(p1))+
    '__'+str(meanpkept0)+'__'+str(meanpkept1)+
    '__'+targs.thefile+'\n')