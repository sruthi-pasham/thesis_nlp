import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from merger import emoji_pattern, train_df, valid_df, test_df, posts, y_test, y_valid, y_train
import re

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


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
            start = round((dim-512)/2)
            new_lst.append(tokenizer(post[start:start+512]))
    return new_lst
            


tokens = tokenize(train_df)
train = {}
keys = ['input_ids', 'token_type_ids', 'attention_mask']
for key in keys:
    train[key] = np.array([torch.tensor(item[key]) for item in tokens])


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

train_vals = jobs_template_dataset(train,y_train)




    






