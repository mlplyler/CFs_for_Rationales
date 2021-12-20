import argparse
from collections import defaultdict

if __name__=='__main__':
  ######## load and parse args
  parser = argparse.ArgumentParser()
  parser.add_argument('tfile')
  parser.add_argument('aspect')
  targs = parser.parse_args()  
  theaspect=int(targs.aspect)

  with open(targs.tfile,'r') as f:
    fstr = f.read()
  flines=fstr.split('\n')
  print('numlines', len(flines))
  ltrack=defaultdict(list)
  for l in flines:
    if '\t' in l:
      afloat = float(l.split('\t')[0].split()[theaspect])
      lab = 1 if afloat>=.5 else 0
      ltrack[lab].append(l)
  lkeys = list(ltrack.keys())
  for k in lkeys:
    print(k,len(ltrack[k]))
  
  newlines=[]
  for i in range(len(ltrack[lkeys[0]])):
    for k in lkeys:
      newlines.append(ltrack[k][i])
  print(len(newlines), len(flines))

  with open(targs.tfile+'_split','w') as f:
    f.write('\n'.join(newlines))
