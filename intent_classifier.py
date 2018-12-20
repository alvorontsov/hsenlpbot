import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import nltk

tokenizer = nltk.tokenize.WordPunctTokenizer()
UNK_IX, PAD_IX = 0,1

# Global max pooling layer
class GlobalMaxPooling(nn.Module):
    def __init__(self, dim=-1):
        super(self.__class__, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return x.max(dim=self.dim)[0]
     
class MessagEncoder(nn.Module):
    def __init__(self, n_tokens, out_size=64):
        """ 
        A simple sequential encoder for titles.
        x -> emb -> conv -> global_max -> relu -> dense
        """
        super(self.__class__, self).__init__()
        self.emb = nn.Embedding(n_tokens, 64, padding_idx=PAD_IX)
        self.conv1 = nn.Conv1d(64, out_size, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(out_size, out_size, kernel_size=3, padding=1)
        self.glob_pool = GlobalMaxPooling()       
        self.dense = nn.Linear(out_size, out_size)
        self.dense2 = nn.Linear(out_size, 2)

    def forward(self, text_ix):
        """
        :param text_ix: int64 Variable of shape [batch_size, max_len]
        :returns: float32 Variable of shape [batch_size, out_size]
        """
        h = self.emb(text_ix)

        # we transpose from [batch, time, units] to [batch, units, time] to fit Conv1d dim order
        h = torch.transpose(h, 1, 2)
        
        # Apply the layers as defined above. Add some ReLUs before dense.
        h = self.conv1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.relu(h)
        h = self.glob_pool(h)
        h = self.dense(h)
        h = self.relu(h)
        logits = self.dense2(h)
        return logits
    
    def predict(self,sentence):
        pred = self.forward(sentence)
        prob = F.softmax(pred)
        #print(prob)
        return int(F.torch.argmax(prob, dim=1))
    
token_to_id = dict()
csv = open("dict.csv","r")
for line in csv:
    a = line.split('#')
    token_to_id[a[0]] = int(a[1])
csv.close()

def as_matrix(sequences, max_len=None):
    """ Convert a list of tokens into a matrix with padding """
    
    if isinstance(sequences[0], str):
        sequences = list(map(str.split, sequences))
        
    max_len = min(max(map(len, sequences)), max_len or float('inf'))
    
    matrix = np.full((len(sequences), max_len), np.int32(PAD_IX))
    for i,seq in enumerate(sequences):
        row_ix = [token_to_id.get(word, UNK_IX) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix
    
    return matrix

model = MessagEncoder(len(token_to_id))
model.load_state_dict(torch.load('model_wghts.pt'))

def feed_model(message):
    message = ' '.join(tokenizer.tokenize(str(message).lower()))
    message = [message]
    mes_rev = torch.LongTensor(as_matrix(message))
    return model.predict(mes_rev)