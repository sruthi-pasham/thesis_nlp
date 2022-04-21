import pandas as pd
import re
import matplotlib.pyplot as plt
import json
import glob
from polyglot.text import Text

with open('./excel/prospects(cs,sd).csv', 'r') as f:
  df_prospects = pd.read_csv(f,sep=';', header=0)
  df_prospects = df_prospects['Program,Job_Prospect'].str.split(',', expand=True)
  df_prospects.rename(columns={0:'Program', 1: 'Job Prospect'} , inplace=True)


pattern = './job_posts/*.json'

#Dump all the json files into a list of posts
posts = []
for file in glob.glob(pattern):
  with open(file,'r') as f:
    json_data = json.loads(f.read())
    posts.append(json_data)

#Removes all the job posts that are not in English
indexes = []
for i, dic in enumerate(posts):
  content = Text(''.join([x for x in dic['description'] if x.isprintable()]))
  if content.language.name != 'English':
    indexes.append(i)

for i in reversed(indexes): posts.pop(i)

#2915 1st version
#2913 2nd version


req_cols = [
 'title'
,'jobId'
, 'skills'
, 'description'
]

df_job_titles= pd.DataFrame(posts, columns=req_cols)

df_job_titles

#csv file for hand annotation
df_job_titles.to_csv('./excel/job_titles_for_annotation.csv')

#data samples for hand annotation
test_dataset = df_job_titles.head(20)
reg1 = '[^a-z-A-Z ].*'
reg2 = '\s\-.*'
full_job_title = [re.sub(reg1,'',st) for st in test_dataset['title']]
job_title = pd.DataFrame([re.sub(reg2,'',st) for st in full_job_title])

#prospects
#strip spaces & convert to lowercase
df_prospects['Program'] = df_prospects['Program'].str.strip().str.lower() 
df_prospects['Job Prospect'] = df_prospects['Job Prospect'].str.strip().str.lower() 


#dropping program and removing duplicates from job prospects, to make unique set of lookup values
df_prospects.drop(['Program'],axis=1, inplace=True)
df_prospects = df_prospects.drop_duplicates()
df_prospects.describe()

print(df_prospects['Job Prospect'])

#job titles
#strip spaces & convert to lowercase
df_j= pd.DataFrame(posts, columns=req_cols)
df_j['title'] = df_j['title'].str.strip().str.lower() 

#drop skill and description
df_j.drop(['skills','description'],axis=1, inplace=True)
df_j.describe()

#making a cross join of prospects & job titles, to be able to normalize 
cartesian_title_x_prospect = df_j.merge(df_prospects, how='cross') 

#creating 'test' column to check if job_prospect is in job title
cartesian_title_x_prospect['test'] = cartesian_title_x_prospect.apply(lambda x: x['Job Prospect'] in x['title'], axis=1)

#converting boolean to string and strip & lower case
cartesian_title_x_prospect['test'] = cartesian_title_x_prospect['test'].astype(str).str.strip().str.lower() 

#normalizing job titles using job prospects and subsequently with default values
cartesian_title_x_prospect['RESULT'] = cartesian_title_x_prospect.apply(lambda x: x['Job Prospect'] if x['Job Prospect'] in x['title'] else     
                                                                              ('robotics engineer' if 'robot' in x['title'] else                                                                      
                                                                              ('golang developer' if 'golang' in x['title'] else                                                                               
                                                                              ('it team leader' if 'team lead' in x['title'] else 
                                                                              ('developer' if 'lead developer' in  x['title'] else 
                                                                              ('power bi developer' if 'power bi' in x['title'] else                                                            
                                                                              ('test manager' if 'test manager' in x['title'] else 
                                                                              ('product owner' if 'owner' in x['title'] else 
                                                                              ('infrastructure consultant' if 'infrastructure' in x['title'] else 
                                                                              ('security engineer' if 'cybersecurity' in x['title'] else                                                                               
                                                                              ('devops engineer' if 'devops' in x['title'] else 
                                                                              ('tester' if 'tester' in x['title'] else 
                                                                              ('embedded developer' if 'embedded' in x['title'] else   
                                                                               ('scrum master' if 'scrum' in x['title'] else 
                                                                              ('ui/ux designer' if 'ux' in ' ' + x['title'] + ' '  or 'user experience' in x['title'] else 
                                                                               ('ui/ux designer' if 'ui' in ' ' + x['title'] + ' ' else 
                                                                                ('it project manager' if 'project manager' in x['title'] else                                                                                                                                                     
                                                                               ('web developer' if 'web' in ' ' + x['title'] + ' ' else 
                                                                               ('digital consultant' if 'digital' in x['title'] else 
                                                                               ('full stack developer' if 'stack' in x['title'] else 
                                                                               ('agile coach' if 'coach' in x['title'] else 
                                                                               ('data scientist' if 'scientist' in x['title'] else 
                                                                               ('security engineer' if 'security' in x['title'] else 
                                                                               ('qa engineer' if 'quality assurance' in x['title'] else 
                                                                               ('developer' if 'node' in ' ' +  x['title'] + ' ' else 
                                                                               ('python developer' if 'python' in x['title'] else 
                                                                                ('machine learning engineer' if 'ml' in ' ' + x['title'] + ' ' else 
                                                                                ('machine learning engineer' if 'machine learning' in ' ' + x['title'] + ' ' else 
                                                                                ('system consultant' if 'systems' in x['title'] or 'system' in x['title'] else 
                                                                                ('front-end developer' if 'front' in x['title'] else 
                                                                                ('backend developer' if 'back' in x['title'] else 
                                                                                ('data analyst' if 'analytics' in x['title'] else                                                                                 
                                                                              ('system administrator' if 'administrator' in x['title'] else 
                                                                              ('tester' if 'testing' in x['title'] else 
                                                                               ('developer' if 'developer' in x['title'] else 
                                                                              ('software architect' if 'architect' in x['title'] else 
                                                                              ('software analyst' if 'analyst' in x['title'] else          
                                                                              ('developer' if 'coding' in x['title'] or 'programmer' in x['title'] else 
                                                                               ('software engineer' if 'engineer' in x['title'] else 
                                                                              ('it Business Analyst' if 'business' in x['title'] else (0)
))))))))))))))))))))))))))))))))))))))), axis=1)

#dropping Job_Prospect column, not relevant anymore
cartesian_title_x_prospect.drop(['Job Prospect'],axis=1, inplace=True)
cartesian_title_x_prospect.describe()


#sort dataframe by jobid and test columns
sorted_df = cartesian_title_x_prospect.sort_values( by = ['jobId','test'])
#sorted_df[(sorted_df.jobId == '520603')]

#take only last row per jobid, to pick the matched row if it exists
final_df= sorted_df.groupby('jobId', as_index=False).nth(-1)


#plot bar with top 15 normalized job titles
top_15_counts = final_df['RESULT'].value_counts()[0:15]
top_15_jobs = ['software engineer', 'developer', 'full stack developer', 'backend developer', 'devops engineer', 'java developer', 'data engineer', 'front-end developer', 
 'site reliability engineer', 'software architect', 'ui/ux designer', 'python developer', 'infrastructure consultant', 'qa engineer', 'security engineer']

plt.figure(figsize=(10,3))

plt.bar(top_15_jobs, top_15_counts, color='black')
plt.xticks(rotation=90)
plt.show()

#Remove jobs with less than 5 counts
#Optimize code!!!
for i in final_df['RESULT'].unique():
  if len(final_df[final_df['RESULT'] == i]) < 5:
    final_df[final_df['RESULT'] == i] = 0


final_df = final_df.sort_index()
pd.DataFrame(final_df['RESULT'].unique()).to_csv(sep='\t', index=False)

final_df = final_df.reset_index()

#need to check these strange (title relevance to IT is very low ) - titles that are still not mapped
strange_titles = final_df[(final_df.RESULT == 0)]
#get the index of the removed job posts
final_df.loc[(final_df.RESULT == 0)].index

#agreed to remove the above strange titles
final_df = final_df[(final_df.RESULT != 0)]

#proportionate stratified sample for each validation and test set
test_df = final_df.groupby('RESULT', group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=123))
final_df = final_df.drop(test_df['jobId'].index)
valid_df = final_df.groupby('RESULT', group_keys=False).apply(lambda x: x.sample(frac=round(len(test_df)/len(final_df), 2), random_state=123)) 
train_df = final_df.drop(valid_df['jobId'].index)

unique_jobs = sorted(train_df['RESULT'].unique())

