import pandas as pd
import nltk
from transformers import BertTokenizer
import numpy as np
import math
import krippendorff
import numpy as np
from nltk.corpus import stopwords
import string
from normalization import test_df, valid_df, train_df, posts
from manual_annotation import pedro_annot, sruthi_annot
from courses import courses
from look_up_table import table
from emojis import emoji_pattern


nltk.download('punkt')
nltk.download('stopwords')


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

sruthi = np.array(sruthi_annot).reshape(1,30*58)
pedro = np.array(pedro_annot).reshape(1,30*58)

tot = np.concatenate((sruthi, pedro), axis=0)
print(krippendorff.alpha(tot))

test_annot = np.array([0 if sruthi_annot[i] != pedro_annot[i] else sruthi_annot[i] for i in range(len(sruthi_annot))])
test = pd.DataFrame(columns=courses, index=test_df['RESULT'].unique(), data=test_annot.reshape(30,58)).to_numpy()

def automatic_annotation(partition):
  arr = np.zeros((len(partition['RESULT']), len(courses)))
  print(arr.shape)
  for num, j in enumerate(partition['RESULT']):
    arr[num, table[j]] = 1
  return arr

def test_annotation():
  arr = np.empty((len(test_df), len(courses)))
  for num, job in enumerate(test_df['RESULT']):
    arr[num] = test[np.where(test_df['RESULT'].unique() == job)]
  return arr

y_test = test_annotation()

y_train = automatic_annotation(train_df).astype(int)
y_valid = automatic_annotation(valid_df).astype(int)

def get_description(partition):
  for i, p in enumerate(posts):
    if p['jobId'] in list(partition['jobId']):
      yield posts[i]['description'][posts[i]['description'].find('Job description'):]
    else:
      continue

def relevant_info(partition):
  lst = []
  iter = get_description(partition)
  
  while True:
    try:
      post = emoji_pattern.sub(r'',next(iter).replace('\n', ' '))
      token = tokenizer(post)

      num_tokens = len(token['input_ids'])
      if 512 - num_tokens < 0:
        start =  math.ceil((num_tokens-512)/2)
        #print(type(', '.join([str(token) for token in token['input_ids'][start:start+512]])))
        lst.append(token['input_ids'][start:start+512])
      else:
        lst.append(token['input_ids'])

    except StopIteration:
      break

  return lst

stop_words = stopwords.words('english')
def cleaner(partition):
  lst = []
  iter = get_description(partition)
  while True:
    try:
      #remove punctuation 
      text = ''.join(['' if char in string.punctuation or char.isdigit() else char.lower() for char in next(iter)])

      #remove emojis
      text = emoji_pattern.sub(r'', text)

      #remove stopwords and first two words of each post ('job' 'description')  
      lst.append(' '.join([w for w in text.split()[2:] if w not in stop_words]))
      
    except StopIteration:
      break
  return lst

train = cleaner(train_df)
valid = cleaner(valid_df)
test = cleaner(test_df)
