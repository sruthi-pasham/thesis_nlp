import numpy as np
from requests import head
import torch
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from merger import emoji_pattern, train_df, valid_df, test_df, posts, y_test, y_valid, y_train, test
from courses import courses
import re
from torch.utils.data import DataLoader
from metrics import binary_classification, empty_arr, metrics
from Bert_analysis import jobs_first, c_matrix1, helper, c_matrix2, balanced_accuracy, coverage, unique_jobs


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
keys = ['input_ids', 'token_type_ids', 'attention_mask']
#tokenizer = BertTokenizer.from_pretrained('./')
#tokenizer.save_pretrained('./')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def bert_cleaner(partition):
    lst = []
    for i, p in enumerate(posts):
        if p['jobId'] in list(partition['jobId']):
            post = posts[i]['description'][posts[i]['description'].find('Job description'):]
            lst.append(re.sub('\s+', ' ', emoji_pattern.sub(r'', post.replace('\n', '')).strip().lower())[len('job description'):])
        else:
            continue
    return lst

def tokenize(old_list, tokenizer, keys):
    new_lst = []
    for post in old_list:
        dim = len(tokenizer(post)['input_ids'])
        if dim < 512:
            new_lst.append(tokenizer(post, padding='max_length', truncation=True, max_length=512))
        else:
            new_lst.append(split_input(post,dim,tokenizer))
    return group_keys(new_lst, keys)


def split_input(post, dim, tokenizer):
    start = round((dim-512)/2)
    token = tokenizer(post)
    for i in token.keys():
        token[i] = token[i][start:start+512]
    return token
           

def group_keys(tokens, keys):
    dic = {}
    for key in keys:
        dic[key] = torch.tensor([item[key] for item in tokens])
    return dic



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

train_dataset = jobs_template_dataset(tokenize(bert_cleaner(train_df), tokenizer, keys), y_train)
valid_dataset = jobs_template_dataset(tokenize(bert_cleaner(valid_df), tokenizer, keys), y_valid)
test_dataset = jobs_template_dataset(tokenize(bert_cleaner(test_df), tokenizer, keys), y_test)


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(courses))#, from_tf=True)
#model = BertForSequenceClassification.from_pretrained('./')
#model.save_pretrained('./')

model.to(device=device)
  
train_loader = DataLoader(train_dataset, batch_size=4)
valid_loader = DataLoader(valid_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

optim = torch.optim.AdamW(model.parameters(), lr=0.01)
epochs = 1
#lr_scheduler = get_linear_schedule_with_warmup(optimizer=optim, num_warmup_steps=0, num_training_steps=epochs)

model.train()
for e in range(epochs):
    for batch in train_loader:
        optim.zero_grad()
        
        inputs = batch['input_ids'].to(device)
        attention = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).to(torch.float16)
        logits = model(inputs, attention).logits
        
        loss = binary_classification(logits, labels)
        loss.backward()
        optim.step()
        #lr_scheduler.step()

    loss_valid = 0
    model.eval()
    for batch in valid_loader:

        with torch.no_grad():
            batch = {k:v.to(device) for k, v in batch.items()}

        loss = binary_classification(model(batch['input_ids'], batch['attention_mask']).logits, batch['labels'].to(torch.float16))
        loss_valid += loss.item()

    print(loss_valid/len(valid_loader))



outcome = []
a, r, f, p = empty_arr()
model.eval()
for batch in test_loader:

    with torch.no_grad():
        batch = {k:v.to(device) for k, v in batch.items()}
    preds = torch.round(torch.sigmoid(model(batch['input_ids'], batch['attention_mask']).logits)).detach().cpu().numpy()

    outcome.append(preds)

    labels = batch['labels'].to(torch.float16).numpy()
    a,r,p,f = metrics(labels, preds, a, r, f, p)
    
print('\nacc: {}\nrecall: {}\nf1: {}\nprecision: {}'.format(np.mean(a) 
                                                ,np.mean(r), np.mean(f), np.mean(p)))
    
    


first_occ = jobs_first()

cm = c_matrix1(y_test, first_occ, helper())

print(cm[1])
print(np.average(cm[1]))

metrics = c_matrix2(y_test.T, helper().T)
print(metrics)

print(unique_jobs)
print(balanced_accuracy())
print(np.average(balanced_accuracy()))

print(coverage(test, cm[3]))
