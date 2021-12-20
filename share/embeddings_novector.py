import numpy as np
import sys
import gzip
import random
import json
from IO import *


def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        print('getting embedding', path)
        for lfi,line in enumerate(fin):
            if type(line)!=str:
                line = line.decode('ascii')
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                yield word

def create_embedding_layer(path,n_d=200,v_max=None):
    embedding_layer = EmbeddingLayerTf(
            n_d = n_d,
            vocab = [ "<unk>", "<padding>","<start>","<end>","<mask>","<one>","<zero>" ],
            embs = load_embedding_iterator(path),
            oov = "<unk>",
            fix_init_embs = False,
            v_max=v_max
        )    
    return(embedding_layer)
    
class EmbeddingLayerTf():
    '''
        Embedding layer that
                (1) maps string tokens into integer IDs
                (2) maps integer IDs into embedding vectors (as matrix)
        Inputs
        ------
        n_d             : dimension of word embeddings; may be over-written if embs
                            is specified
        vocab           : an iterator of string tokens; the layer will allocate an ID
                            and a vector for each token in it
        oov             : out-of-vocabulary token
        embs            : an iterator of (word, vector) pairs; these will be added to
                            the layer
        fix_init_embs   : whether to fix the initial word vectors loaded from embs
        
        tensorflow implementation:
            NB using same notation as github page of Tao Lei
    '''

    def __init__(self, n_d, vocab, oov="<unk>", embs=None, fix_init_embs=True,v_max=None):
        if embs is not None:
          lst_words = [ ]     
          vocab_map = {}                
          for word in vocab:
            vocab_map[word]=len(vocab_map)
            lst_words.append(word)
          for word in embs: 
            assert word not in vocab_map, "Duplicate words in initial embeddings"
            if 1: 
              vocab_map[word] = len(vocab_map)
              lst_words.append(word)            
              if v_max is not None and len(vocab_map)>=v_max:
                break
            else:
              print('wierd embedding len!', word, len(vector))
        self.lst_words = lst_words  
        self.vocab_map = vocab_map
        # out of vocabulary words
        if oov is not None and oov is not False:            
            # out of vocab word must be in vocab
            assert oov in self.vocab_map, "oov {} not in vocab".format(oov)
            self.oov_tok = oov
            self.oov_id = self.vocab_map[oov]
        else:            
            # otherwise default id
            self.oov_tok = None
            self.oov_id = -1
        self.n_V = len(self.vocab_map)
        print('\n', 'VOCAB SIZE', self.n_V,'\n')        
        self.padding_id = self.vocab_map['<padding>']

    def map_to_words(self, ids):            
        n_V, lst_words = self.n_V, self.lst_words
        return([ lst_words[i] if i < n_V else "<err>" for i in ids ])
        
        
    def map_to_ids(self, words, filter_oov=False):
        '''
            map the list of string tokens into a numpy array of integer IDs
            Inputs
            ------
            words           : the list of string tokens
            filter_oov      : whether to remove oov tokens in the returned array
            Outputs
            -------
            return the numpy array of word IDs
        '''
        
        vocab_map = self.vocab_map
        oov_id = self.oov_id
        if filter_oov:
            not_oov = lambda x: x!=oov_id
            return(np.array(
                    filter(not_oov, [ vocab_map.get(x, oov_id) for x in words ]),
                    dtype="int32"
                ))
        else:
            return(np.array(
                    [ vocab_map.get(x, oov_id) for x in words ],
                    dtype="int32"
                ))
 
  
