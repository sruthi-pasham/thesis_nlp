import pandas as pd
from courses import courses
from normalization import unique_jobs

print(unique_jobs)

m1_df = pd.read_excel('./excel/matrices.xlsx', sheet_name='matrix1', index_col=None, header=None)
m2_df = pd.read_excel('./excel/matrices.xlsx', sheet_name='matrix2', index_col=None, header=None)

#convert to numpy array
m1 = m1_df.to_numpy()

tot = []
for i in range(m1.shape[0]):
  tmp = []
  for j in range(m1.shape[1]):
    tmp.append(str(m1[i][j]).split(', '))
  tot.append(tmp)

def index_courses(s):
  return [courses.index(i) for i in s if i != 'nan']

table = {}
def create_dictionary(arr, job):
  s = []
  for i in arr:
    if i[1] > 0:
      s.append(str(m2_df.T[i[0]][i[1]]).split(', '))
      s.append(str(m2_df.T[i[0]][i[1]-1]).split(', '))
    else:
      s.append(str(m2_df.T[i[0]][i[1]]).split(', '))
  s = set([j for i in s for j in i])
  table[job] = index_courses(s)

#part1 job title position table1
for job in unique_jobs:
  x = [i for x in tot for i in x if job in i]
  results = []
  for item in x:
    if 'nan' in item:
      continue
    y = [i for i in tot if item in i][0]
    results.append((tot.index(y), y.index(item)))
  create_dictionary(results, job)

print(table)