import numpy as np
import sys
import gzip
import random
import json
from IO import *


def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        for lii,line in enumerate(fin):
            if type(line)!=str:
                line = line.decode('ascii')
            line = line.strip() 
            if line:
                parts = line.split()
                word = parts[0]
                vals = np.array([ float(x) for x in parts[1:] ])
                yield word, vals

def create_embedding_layer(path,n_d=200):
    embedding_layer = EmbeddingLayerTf(
            n_d = n_d,
            vocab = [ "<unk>", "<padding>","<start>","<end>","<mask>","<one>","<zero>" ],
            embs = load_embedding_iterator(path),
            oov = "<unk>",
            fix_init_embs = False
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

    def __init__(self, n_d, vocab, oov="<unk>", embs=None, fix_init_embs=True):        
        if embs is not None:

          lst_words = [ ]     
          vocab_map = {}      
          emb_vals = [ ]      

          for word in vocab:
            if word not in vocab_map:                                            
              vocab_map[word] = len(vocab_map)
              emb_vals.append(random_init((n_d,))*(0.001 if word != oov else 0.0))
              lst_words.append(word)
              
          for wi,(word, vector) in enumerate(embs):
            assert word not in vocab_map, "Duplicate words in initial embeddings"
            if len(vector)!=n_d:
                print('wierd embedding len!', word, len(vector),wi) 
                if len(vector)>n_d:
                    vector=vector[:n_d]
                elif len(vector)<n_d:
                    vector = np.append(vector,
                                  np.zeros(n_d-len(vector)),axis=0)
            vocab_map[word] = len(vocab_map)
            emb_vals.append(vector)
            lst_words.append(word)
            self.init_end = len(emb_vals) if fix_init_embs else -1 
                
          # if using other word vectors and the size isn't correct, correct length
          print('embedding stuff', n_d, np.shape(emb_vals))
          if n_d != len(emb_vals[0]):
            n_d = len(emb_vals[0])


          emb_vals = np.vstack(emb_vals)
          self.vocab_map = vocab_map
          self.lst_words = lst_words
          self.set_words = set(lst_words)  
        else:
                
          # otherwise randomly initialize the word vectors
          lst_words = [ ]
          vocab_map = {}
          for word in vocab:
            if word not in vocab_map:
              vocab_map[word] = len(vocab_map)
              lst_words.append(word)

          self.lst_words = lst_words
          self.vocab_map = vocab_map
          emb_vals = random_init((len(self.vocab_map), n_d))
          self.init_end = -1  # set it so the word vectors can be updated
                
            
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
            
        self.embeddings = emb_vals
        if self.init_end > -1:
            self.embeddings_trainable = self.embeddings[self.init_end:]
        else:
            self.embeddings_trainable = self.embeddings
        
        # set dimensions
        self.n_V = len(self.vocab_map)
        self.n_d = n_d

    def map_to_words(self, ids):
        # trivial
            
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

    def call(self,x):
        '''
            Fetch and return the word embeddings given word IDs x
            Inputs
            ------
            x           : a theano array of integer IDs
            Outputs
            -------
            a a numpy matrix of word embeddings.
            
        '''
        return(self.embeddings[x])
    
    @property
    def params(self):
        return([ self.embeddings_trainable ])

    @params.setter
    def params(self, param_list):
        self.embeddings.set_value(param_list[0].get_value())            
            
    def map_to_words(self, ids):
        # trivial            
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

    def forward(self, x):      
        return(self.embeddings[x])
        
    
    @property
    def params(self):
        return([ self.embeddings_trainable ])

    @params.setter
    def params(self, param_list):
        self.embeddings.set_value(param_list[0].get_value())
