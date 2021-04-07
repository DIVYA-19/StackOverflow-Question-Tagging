import numpy as np
import torch
from torch import nn
from transformers import BertModel

import preprocessing as P 

threshold = 0.2
tags = eval(open('data/tags.txt', 'r').read())
tags = np.array(list(tags))

def init_weights(m):
    """ initial weight intialization"""
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier,self).__init__()
        # Specify hidden size of Bert, hidden size of our classifier, and number of labels
        D_in, H, H2, H3, D_out = 768, 1024, 512, 256, tags.shape[0]
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")        
        self.out = nn.Sequential(nn.Dropout(0.1), nn.ReLU(), nn.Linear(D_in, H), nn.Dropout(0.3), nn.ReLU(), nn.Linear(H, D_out))
        self.out.apply(init_weights)

    def forward(self,input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:,0,:]
        
        out = self.out(last_hidden_state_cls)
        
        return out

model = BertClassifier().to("cpu")
model.load_state_dict(torch.load("model/bert_classifier.model", map_location=torch.device('cpu')))
model.eval()

def predict(texts):
    """
    for the given texts returns tag names in dictionary format

    param: texts
    type: List[List of titles, List of bodies]
    return: dictionary of predictions
    """
    predictions = {}
    titles = texts[0]
    bodies = texts[1]
    ids, atts = P.embedding(titles, bodies)

    logits = model(ids, atts)
    preds = logits.sigmoid().detach().cpu().numpy()
    preds= np.array( preds > threshold)

    for i in range(len(titles)):
        indxs = np.where(preds[i] == True)
        predictions[i] = list(tags[indxs])

    return predictions
