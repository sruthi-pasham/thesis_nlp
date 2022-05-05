import torch
import transformers
import numpy as np
from courses import courses
from merger import emoji_pattern, train_df, valid_df, test_df, posts, y_test, y_valid, y_train
import re
from BERT import group_keys, tokenize, split_input, jobs_template_dataset, binary_classification, metrics, empty_arr, keys
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassifcation



tokenizer = AutoTokenizer.from_pretrained('jjzha/jobbert-base-cased')

def jobBERT_cleaner(partition):
    lst = []
    for i, p in enumerate(posts):
        if p['jobId'] in list(partition['jobId']):
            post = posts[i]['description'][posts[i]['description'].find('Job description'):]
            lst.append(re.sub('\s+', ' ', emoji_pattern.sub(r'', post.replace('\n', '')).strip().upper())[len('job description'):])
        else:
            continue
    return lst

train_dataset = jobs_template_dataset(tokenize(jobBERT_cleaner(train_df), tokenizer, keys), y_valid)
valid_dataset = jobs_template_dataset(tokenize(jobBERT_cleaner(valid_df), tokenizer, keys), y_valid)
test_dataset = jobs_template_dataset(tokenize(jobBERT_cleaner(test_df), tokenizer, keys), y_test)

model = AutoModelForSequenceClassifcation.from_pretrained('jjzha/jobbert-base-cased')

train_loader = DataLoader(train_dataset, batch_size=16)
valid_loader = DataLoader(valid_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

optim = torch.optim.AdamW(model.parameters(), lr=0.0001)

a, r, f, p = empty_arr()
model.train()
for batch in train_loader:
    optim.zero_grad()

    inputs = batch['input_ids']
    attention = batch['attention_mask']
    labels = batch['labels'].to(torch.float16)
    logits = model(inputs, attention).logits

    loss = binary_classification(logits, labels)

    print(loss)

    preds = torch.round(torch.sigmoid(logits)).detach().numpy()
    target = labels.numpy()
    a,r,p,f = metrics(target, preds, a, r, f, p)

print('epoch: 1 \nacc: {}\nrecall: {}\nf1: {}\nprecision: {}'.format(np.mean(a)
                                                ,np.mean(r), np.mean(f), np.mean(p)))

a, r, f, p = empty_arr()
model.eval()
for i in test_loader:
    inputs = batch['input_ids']
    attention = batch['attention_mask']
    labels = batch['labels'].to(torch.float16).numpy()
    preds = torch.round(torch.sigmoid(model(inputs, attention).logits)).detach().numpy()

    a,r,p,f = metrics(labels, preds, a, r, f, p)

print('\nacc: {}\nrecall: {}\nf1: {}\nprecision: {}'.format(np.mean(a)
                                                ,np.mean(r), np.mean(f), np.mean(p)))



