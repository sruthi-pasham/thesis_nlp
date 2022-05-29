import torch
from torch import nn
import transformers
import numpy as np
from courses import courses
from merger import emoji_pattern, train_df, valid_df, test_df, posts, y_test, y_valid, y_train, test
import re
from BERT import tokenize, jobs_template_dataset, keys
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassifcation, get_linear_schedule_with_warmup
from metrics import metrics, binary_classification, empty_arr
from Bert_analysis import jobs_first, c_matrix1, helper, c_matrix2, balanced_accuracy, coverage, unique_jobs
from sklearn.metrics import confusion_matrix


tokenizer = AutoTokenizer.from_pretrained('jjzha/jobbert-base-cased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def jobBERT_cleaner(partition):
    lst = []
    for i, p in enumerate(posts):
        if p['jobId'] in list(partition['jobId']):
            post = posts[i]['description'][posts[i]['description'].find('Job description'):]
            lst.append(re.sub('\s+', ' ', emoji_pattern.sub(r'', post.replace('\n', '')).strip().upper())[len('job description'):])
        else:
            continue
    return lst

train_dataset = jobs_template_dataset(tokenize(jobBERT_cleaner(train_df), tokenizer, keys), y_train)
valid_dataset = jobs_template_dataset(tokenize(jobBERT_cleaner(valid_df), tokenizer, keys), y_valid)
test_dataset = jobs_template_dataset(tokenize(jobBERT_cleaner(test_df), tokenizer, keys), y_test)

class BertLayer(nn.Module):
    def __init__(self, bert):
        super(BertLayer, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 58)
    

    def forward(self, input, attention):

        out = self.bert(input, attention)[1]
        out = self.dropout(out)

        logits = self.fc(out)

        return logits

#Freeze BERT parameters
#model = BertModel.from_pretrained('jjzha/jobbert-base-cased')
#for param in model.parameters():
    #param.requires_grad = False

#model = BertLayer(model)

model = AutoModelForSequenceClassifcation.from_pretrained('jjzha/jobbert-base-cased', num_labels=len(courses))

train_loader = DataLoader(train_dataset, batch_size=8)
valid_loader = DataLoader(valid_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

model.to(device=device)

optim = torch.optim.AdamW(model.parameters(), lr=0.01)
#lr_scheduler = get_linear_schedule_with_warmup(optimizer=optim, num_warmup_steps=0, num_training_steps=epochs)
epochs = 1

for e in range(epochs):
    model.train()
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
