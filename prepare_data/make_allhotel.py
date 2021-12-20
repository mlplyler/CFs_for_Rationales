import os
import numpy as np
import json
import spacy

'''
numfiles 12773
totreviews 1621956
set reviews 1178548
'''

thefol = './data/TripAdvisorJson/json/'

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
# Create a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nlp.vocab)

# Construction 2
from spacy.lang.en import English
nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.Defaults.create_tokenizer(nlp)

### load all the data
numtf = len(os.listdir(thefol))
print('numfiles',numtf)
totreviews=0
#allcontent=set()
savename = './data/allcontent.txt'
with open(savename,'w') as fsave:
    for thef in os.listdir(thefol):
        with open(thefol+thef,'r') as f:
            fstr = f.read()
            moo = json.loads(fstr)
            totreviews+=len(moo['Reviews'])
            newdocs=[' '.join([t.text for t in nlp(m['Content'])]).lower()
                        for m in moo['Reviews']]                     
            dumpstr = '\n'.join(newdocs)+'\n'            
            fsave.write(dumpstr)            

print('totreviews', totreviews)

with open('./data/allcontent.txt','r') as f:
    fstr = f.read()
flines=fstr.split('\n')
print('flines',len(flines))

setlines = set(flines)
print('set lines', len(setlines))
dumpstr = '\n'.join(setlines)
with open('./setcontent.txt','w') as f:
    f.write(dumpstr)
## load dev and annotate

## load dev and annotate text
devannotext=[]

fnames = [
        './data/hotel_Location.pred_att.gold_att.train'      
]
for fname in fnames:
    with open(fname,'r') as f:
        fstr = f.read()
    flines = fstr.split('\n')[1:-1]
    devannotext+=[f.split('\t')[2] for f in  flines]

fnames = [
        './data/location.train',
        './data/location.dev'
            ]


for fname in fnames:
    with open(fname,'r') as f:
        fstr = f.read()
    flines = fstr.split('\n')[:-1]
    devannotext+=[f.split('\t')[1] for f in  flines]
print('devannotext',len(devannotext))
devannoset = set(devannotext)
print('devannoset',len(devannoset))

## trim to make it comparable
trimmed = set(map(lambda x:x.replace(' ',''), devannoset))
print(type(trimmed),len(trimmed))

'''
when you match on text exactly, you miss 2,
the first is due to a triple space vs single space
the second is due to a double vs single space

we can conclude that they probably used the same 
or very similar tokenizer, but we should match based on trim to sure
'''

# load set content
with open('./data/setcontent.txt','r') as f:
    fstr = f.read()
allr=fstr.split('\n')
print('allr',len(allr))

## do it by line
numinter = 0
matchlines=[]
setmatch=set()
with open('./data/setcontent.txt','r') as f:
    for li,line in enumerate(f):
      fline = line.replace('\n','')#.replace(' ','')
      fline2 = line.replace('\n','').replace(' ','')
      if fline not in devannoset and fline2 in trimmed:
          print('BAD one!!')
          print('fline')
          print(fline)
      if fline in devannoset:         
          numinter+=1
          matchlines.append(li)
          setmatch.add(fline)
      if li%100000==0:
          print(li,numinter)



## these are the ones we didint find
notmatched = devannoset.difference(setmatch)
print(len(notmatched))


numinter = 0
matchlines=[]
setmatch=set()
with open('./data/allhotel_noloc.txt','w') as fdump:
    with open('./data/setcontent.txt','r') as f:
        for li,line in enumerate(f):
            fline2 = line.replace('\n','').replace(' ','')

            if fline2 in trimmed:         
                numinter+=1
                matchlines.append(li)
                setmatch.add(fline2)
            else:
                fdump.write(line)
            if li%100000==0:
                print(li,numinter)

print('matchedlines', len(matchlines))

notmatched = trimmed.difference(setmatch)
print('notmatched', notmatched)

## make it rat
numlines=0
with open('./data/allhotel_noloc.ratform','w') as fdump:
    with open('./data/allhotel_noloc.txt','r') as f:
        for line in f:
            newline = '69\t'+line
            fdump.write(newline)
            numlines+=1
print(numlines)

## split into train dev
import random
numdev = 5000
devi = set(random.sample(range(numlines),numdev))
print(len(devi))

with open('./data/allhotel_noloc.train.ratform','w') as fdump:
    with open('./data/allhotel_noloc.dev.ratform','w') as fdump2:
        with open('./data/allhotel_noloc.ratform','r') as f:
            for i,line in enumerate(f):
                if i in devi:
                    fdump2.write(line)
                else:
                    fdump.write(line)
