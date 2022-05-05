import transformers
import numpy as np
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from courses import courses
from merger import emoji_pattern, train_df, valid_df, test_df, posts, y_test, y_valid, y_train
import re
from BERT import group_keys, tokenize, split_input, jobs_template_dataset, binary_classification, metrics, empty_arr
from torch.utils.data import DataLoader

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
keys = ['input_ids', 'attention_mask']

#create Roberta cleaner
def roberta_cleaner(partition):
    lst = []
    for i, p in enumerate(posts):
        if p['jobId'] in list(partition['jobId']):
            post = posts[i]['description'][posts[i]['description'].find('Job description'):]
            lst.append(re.sub('\s+', ' ', emoji_pattern.sub(r'', post.replace('\n', '')).strip())[len('job description'):])
        else:
            continue
    return lst


train_dataset = jobs_template_dataset(tokenize(roberta_cleaner(train_df), tokenizer, keys), y_valid)
valid_dataset = jobs_template_dataset(tokenize(roberta_cleaner(valid_df), tokenizer, keys), y_valid)


model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(courses), from_tf=True)
model.save_pretrained('./')


train_loader = DataLoader(train_dataset, batch_size=16)
valid_loader = DataLoader(valid_dataset, batch_size=16)
