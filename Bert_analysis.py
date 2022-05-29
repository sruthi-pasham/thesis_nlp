from normalization import test_df

def outcome_gen(arr):
  for i in range(len(outcome)):  
    yield arr[i]



def helper():
  n = outcome_gen(outcome)
  a = []
  while True:
    try:
      for i in next(n):
        a.append(i)
    except StopIteration:
      return np.array(a)


unique_jobs =  test_df['RESULT'].unique()

def jobs_first():
    first_occ = []
    df = test_df.reset_index()#.to_numpy()
    for j in unique_jobs:
        first_occ.append(df.loc[df['RESULT'] == j].index[0])
    return first_occ

#first_occ = jobs_first()

def c_matrix1(test, lst, preds):
    f_arr = []
    for tup in [(0,0),(0,1),(1,0),(1,1)]:
        helper_arr = []
        for i, p in enumerate(preds):
          helper_arr.append(confusion_matrix(test[i], p)[tup[0]][tup[1]])
        for num in range(len(lst)-1): 
          f_arr.append(round(sum(helper_arr[lst[num]:lst[num+1]])/len(helper_arr[lst[num]:lst[num+1]]), 2))
          if num+1== len(lst)-1:
            f_arr.append(round(sum(helper_arr[lst[num+1]:])/len(helper_arr[lst[num+1]:]),2))
    return np.array(f_arr).reshape((4,30))

#cm = c_matrix1(y_test, first_occ, helper())

#print(cm[1])
#print(np.average(cm[1]))
#print(metrics[3])

def c_matrix2(test, preds):
    arr = []
    removed_courses = []
    for tup in [(0,0),(0,1),(1,0),(1,1)]:
        for i, p in enumerate(preds):
            if np.sum(p) + np.sum(test[i]) == 0:
                removed_courses.append(i)
                continue
            arr.append(confusion_matrix(test[i], p)[tup[0]][tup[1]])
    return np.array(arr).reshape((4,46))

#metrics = c_matrix2(y_test.T,helper().T)

#print(metrics)

def balanced_accuracy():
    ratio = []
    for i in range(46):
        tpr = (metrics[3][i])/(metrics[2][i]+metrics[3][i]) 
        tnr = (metrics[0][i])/(metrics[1][i]+metrics[0][i])
        ratio.append(round((tpr+tnr)/2,2))
    return ratio

#print(unique_jobs)
#print(balanced_accuracy())
#print(np.average(balanced_accuracy()))
#tn, fp, fn, tp

#which courses appear the most, right or wrong
#positive likelihood ratio = (TP/FN+TP)/(FP/TN+FP)

#% of courses covered by each job title
def coverage(y_test, tp):
    return [round(tp[i]/np.sum(y_test[i]),2) for i in range(len(y_test))]

#print(coverage(test, cm[3]))