import matplotlib.pyplot as plt
from math import sqrt,pow
import pandas as pd
import numpy as np
from Queue import Queue

#READING DATA AND INITILAZING
f, (ax1, ax2) = plt.subplots(1, 2)
data = pd.read_csv('hw05_data_set.csv')
data=data.reindex(np.random.permutation(data.index))
N_train=100
N_test=33

train_data= data[0:N_train]
test_data=data[N_train:]

Ps=[]
errors=[]

train_values=[]
for index,row in train_data.iterrows():
    train_values.append([row['x'],row['y']])

test_values=[]
for index,row in test_data.iterrows():
    test_values.append([row['x'],row['y']])

#ITERATIVE DECISION TREE ALGORITHM
def DecisionTreeRegression(P,data,indices,queue):
    if(not len(data)<=P):

        #FIND THE INDEX TO DIVIDE INTO TWO CATEGTORIES: note that 1 one dimensional case average of the training is the best split point!
        sum=0
        for i in data:
            sum+=i[0]
        index=sum*1.0/len(data)
        indices.append(index)

        #DIVIDE INTO TWO CATEGORIES:
        S1=[]
        S2=[]

        for i in data:
            if(i[0]<=index):
                S1.append([i[0],i[1]])
            else:
                S2.append([i[0],i[1]])
        if(len(S2)==0):
            S1=np.array_split(S1,2)[0]
            S2=np.array_split(S1,2)[1]
        #ADD TO QUEUE:
        if(len(S1)>P):
            queue.put(S1)
        if(len(S2)>P):
            queue.put(S2)


#FOR LOOP FOR DIFFERENT P VALUES AND PLOTTING AT P=10
for P in range(1, 21):
    globalQueue = Queue()
    globalQueue.put(train_values)
    globalIndices=[]
    print P
    Ps.append(P)
    while (not globalQueue.empty()):
        popped=globalQueue.get()
        DecisionTreeRegression(P, popped, globalIndices,globalQueue)
    error=0
    globalIndices.append(0)
    globals = sorted(globalIndices)
    print "x1 Prune Indices: " + str(globals)
    dic = {}

    for i in range(0, len(globals)):
        counter = 0
        sum = 0
        if (i == len(globals) - 1):
            for index, data in train_data.iterrows():
                if (data['x'] < 60 and data['x'] >= globals[i]):
                    sum += data['y']
                    counter += 1
        else:
            for index, data in train_data.iterrows():
                if (data['x'] < globals[i + 1] and data['x'] >= globals[i]):
                    sum += data['y']
                    counter += 1
        if (counter != 0):
            dic[globals[i]] = sum * 1.0 / counter
        else:
            dic[globals[i]] = dic[globals[i - 1]]

    if (P == 10):
        xs = [0]
        ys = []
        for i in sorted(dic):
            if (i != 0):
                xs.append(i)
                xs.append(i)
            ys.append(dic[i])
            ys.append(dic[i])

        xs.append(60)
        ax1.plot(train_data['x'], train_data['y'], 'b.',label='training')
        ax1.plot(test_data['x'], test_data['y'], 'r.',label='test')
        ax1.plot(xs, ys, 'k-')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('P = 10')
        ax1.legend()

    for indexer, row in test_data.iterrows():

        for i in range(0, len(globals)):
            averageY = dic[globals[i]]
            if (i == len(globals) - 1):
                if (row['x'] < 60 and row['x'] >= globals[i]):
                    error = error + pow((averageY - row['y']), 2)
            else:
                if (row['x'] < globals[i + 1] and row['x'] >= globals[i]):
                    error = error + pow((averageY - row['y']), 2)

    error = sqrt(error*2 / len(train_values))
    errors.append(error)

ax2.plot(Ps, errors, 'k-')
ax2.plot(Ps, errors, 'k.')
ax2.set_xlabel('P')
ax2.set_ylabel('RMSE')
plt.show()