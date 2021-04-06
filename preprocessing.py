import re
import pandas as pd
import numpy as np

import torch
from transformers import BertTokenizer, DistilBertModel

stop_words = tags = eval(open('data/stop_words.txt', 'r').read())
stop_words = list(set(map(lambda x:x.lower(), stop_words)))

title_max_length = 10
body_max_length = 100

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[/?{}|,()`<>[]"]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)

    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)
    # text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = text.split(' ')

    return text

def encode(texts, text_type='title'):
    ''' returns encoded ids and attentionmasks for the given texts '''
    if text_type == 'title':
        max_length = title_max_length
    else:
        max_length = body_max_length
    
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = "bert-base-uncased")
    
    ids = []
    att_masks = []
    for text in texts:
        text_ids = tokenizer.encode_plus(text, max_length=max_length, padding="max_length", truncation=True)
        ids.append(text_ids.input_ids)
        att_masks.append(text_ids.attention_mask) 
    
    ids = torch.tensor(ids)
    att_masks = torch.tensor(att_masks)

    return ids, att_masks


def text_preprocessing(text):
    """ Cleaning and parsing the text """
    nopunc = clean_text(text)
    remove_stopwords = [w for w in nopunc if w not in stop_words]
    combined_text = ' '.join(remove_stopwords)
    return combined_text


def embedding(titles, bodies):
    """ returns the encoded texts in model input format """

    titles = pd.Series(titles)
    bodies = pd.Series(bodies)
    preprocessed_titles = titles.apply(lambda x: text_preprocessing(x))
    preprocessed_bodies = bodies.apply(lambda x: text_preprocessing(x))
    embeddings_titles_ids, embeddings_titles_att_masks= encode(preprocessed_titles)
    embeddings_bodies_ids, embeddings_bodies_att_masks = encode(preprocessed_bodies)

    return torch.cat((embeddings_titles_ids, embeddings_bodies_ids), dim=1), torch.cat((embeddings_titles_att_masks, embeddings_bodies_att_masks), dim=1)

