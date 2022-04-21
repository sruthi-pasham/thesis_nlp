from numpy import round_
import torch
from transformers import BertTokenizer, BertModel
from merger import emoji_pattern, train_df, valid_df, test_df, posts
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
            
tokenize(train_df)


    






