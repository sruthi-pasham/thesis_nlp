import numpy as np
import transformers
import torch
from transformers import AdamW, BertTokenizer, BertForSequenceClassification
from merger import emoji_pattern, train_df, valid_df, test_df, posts, y_test, y_valid, y_train
from courses import courses
import re
from torch import nn
from torch.utils.data import DataLoader



tokenizer = BertTokenizer.from_pretrained('./')
#tokenizer.save_pretrained('./')
torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def bert_cleaner(partition):
    lst = []
    for i, p in enumerate(posts):
        if p['jobId'] in list(partition['jobId']):
            post = posts[i]['description'][posts[i]['description'].find('Job description'):]
            lst.append(re.sub('\s+', ' ', emoji_pattern.sub(r'', post.replace('\n', '')).strip().lower())[len('job description'):])
        else:
            continue
    return lst



def tokenize(partition):
    old_lst = bert_cleaner(partition)
    new_lst = []
    for post in old_lst:
        dim = len(tokenizer(post)['input_ids'])
        if dim < 512:
            new_lst.append(tokenizer(post, padding='max_length', truncation=True, max_length=512))
        else:
            new_lst.append(split_input(post,dim))
    return new_lst


def split_input(post, dim):
    start = round((dim-512)/2)
    token = tokenizer(post)
    for i in token.keys():
        token[i] = token[i][start:start+512]
    return token
           

#turn this into a function
tokens = tokenize(train_df)
valid = {}
keys = ['input_ids', 'token_type_ids', 'attention_mask']
for key in keys:
    valid[key] = np.array([torch.tensor(item[key]) for item in tokens])




class jobs_template_dataset(torch.utils.data.Dataset):
  
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels
  
  def __getitem__(self, idx):
    items = {key:val[idx].clone().detach() for key, val in self.encodings.items()}
    items['labels'] = torch.tensor(self.labels[idx])
    return items
  
  def __len__(self):
    return len(self.labels)


valid_dataset = jobs_template_dataset(valid,y_valid)

#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(courses), from_tf=True)
model = BertForSequenceClassification.from_pretrained('./')
#model.save_pretrained('./')

#model = model.to(device='device')

def binary_classification(inputs, targets):
    criterion = nn.BCEWithCrossEntropyLoss()
    return criterion(inputs, targets)


valid_loader = DataLoader(valid_dataset, batch_size=32)

optim = AdamW(model.parameters(), lr=0.0001)


model.train()
for batch in valid_loader:
    optim.zero_grad()
    
    inputs = batch['input_ids']
    attention = batch['attention_mask']
    labels = batch['labels']

    logits = model(inputs, attention)

    loss = binary_classification(logits, labels)

    loss.backward()
    optim.step()





