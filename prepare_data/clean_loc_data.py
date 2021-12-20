import numpy as np
from collections import Counter

## things to keep out of train
fnames = [
          './data/hotel_Location.pred_att.gold_att.train',          
          './data/hotel_Location.dev',          
          ]
text=[];wl=[];y=[];lens=[];
for fname in fnames:
    with open(fname,'r') as f:
        fstr = f.read()
    flines = fstr.split('\n')[:-1]
    print(fname, len(flines))    
    print('flines', len(flines))
    text+=[f.split('\t')[2] for f in  flines]
    print('text')
    
gold_text = list(text)   
set_gold = set(gold_text)

## place to get train rows from
fnames = [
          './data/hotel_Location.train',    
            ]

text=[];wl=[];y=[];lens=[];skipped=0;
for fname in fnames:
    with open(fname,'r') as f:
        fstr = f.read()
    flines = fstr.split('\n')[1:-1]
    print(fname, len(flines))
    for f in flines:
        t = f.split('\t')[2]
        if t not in set_gold and len(t)>5:
            text.append(t)
            wl.append(t.split(' '))
            y.append(f.split('\t')[1])
            lens.append(len(wl[-1]))
            
        else:
            skipped+=1
            
print('num docs', len(lens))         
print('skipped', skipped)
ccount = Counter(y)
print(ccount)            


## bring balance to the classes
minv = min(ccount.values())
ccount2 = {'0':0,'1':0}
text2=[];y2=[];
for i in range(len(text)):
    if ccount2[y[i]]<minv:
        ccount2[y[i]]+=1
        text2.append(text[i])
        y2.append(y[i])
print(ccount2)
print(Counter(y2))


## save it
savename = './data/location.train'
dumpstr = '\n'.join([y2[i]+'\t'+text2[i] for i in range(len(text2))])
with open(savename,'w') as f:
    f.write(dumpstr)


## double check
with open(savename,'r') as f:
    fstr=f.read()
flines=fstr.split('\n')
print('flines', len(flines))
y3 = [f[0] for f in flines]
print(Counter(y3))
        