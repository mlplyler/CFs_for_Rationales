import numpy as np
from collections import Counter

savename = './data/jw.allhotel_noloc.txt'

fnames = ['./data/allhotel_noloc.train.ratform']

## get text
text=[];wl=[];y=[];lens=[];
for fname in fnames:
    with open(fname,'r') as f:
        fstr = f.read()
    flines = fstr.split('\n')[1:-1]
    print('flines', len(flines))
    text+=[f.split('\t')[1] for f in  flines]  ##[-1]
flines=None    

## count words
counts= Counter()
for t in text:
    counts.update(t.split())
print('numcounts', len(counts))

## sort words by count
allstrs = [k[0] for k in counts.most_common()]
print(allstrs[:20])
print('numstr',len(allstrs))

## save it
mystr = '\n'.join(allstrs)
with open(savename, 'w') as f:
    f.write(mystr)