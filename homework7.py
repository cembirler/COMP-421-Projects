import random
import pandas as pd
from math import e,pow, sqrt
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

random.seed(40)
np.random.seed(50)

f,(ax1, ax2) = plt.subplots(1, 2)

mean = [[2.5, 2.5],
        [-2.5,2.5],
        [-2.5,-2.5],
        [2.5,-2.5],
        [0.0,0.0]]
cov=[[[0.8,-0.6],[-0.6,0.8]],
     [[0.8, 0.6], [0.6, 0.8]],
     [[0.8, -0.6], [-0.6, 0.8]],
     [[0.8, 0.6], [0.6, 0.8]],
     [[1.6, 0.0], [0.0, 1.6]]]

n=[50,50,50,50,100]

X0= np.random.multivariate_normal(mean[0], cov[0], n[0])
X1= np.random.multivariate_normal(mean[1], cov[1], n[1])
X2= np.random.multivariate_normal(mean[2], cov[2], n[2])
X3= np.random.multivariate_normal(mean[3], cov[3], n[3])
X4= np.random.multivariate_normal(mean[4], cov[4], n[4])

X=[]
for i in X0:
    X.append(i)
for i in X1:
    X.append(i)
for i in X2:
    X.append(i)
for i in X3:
    X.append(i)
for i in X4:
    X.append(i)


clean_data = [[item for item in row] for row in X]
frame=pd.DataFrame(clean_data, columns=list('xy'))

#choose random 5-means:
V=[]
for i in range(0,5):
    tempMean=[random.random()*6-3,random.random()*6-3]
    V.append(tempMean)

#5-means clustering 2 iterations:
vs='a'
for i in range(0,2):
    labels=[]
    for indexer, row in frame.iterrows():
        minDistance=999
        bestLabel=-1
        for j in range(0,5):
            tempDistance=sqrt((row[0]-V[j][0])*(row[0]-V[j][0])+(row[1]-V[j][1])*(row[1]-V[j][1]))
            if(tempDistance<minDistance):
                minDistance=tempDistance
                bestLabel=j
        labels.append(bestLabel)
    #print len(labels)
    frame['label']=labels
    vs=frame.groupby('label').mean()
    vs.reset_index(level=0, inplace=True)

    for i in range(0, 5):
        if i in vs['label']:
            V[i] = [vs['x'][i],vs['y'][i]]

initialCovs = []

XXX=[[],[],[],[],[]]
for indexer, row in frame.iterrows():
    XXX[(int)(row['label'])].append([row['x'],row['y']])
for i in range(0,5):
    initialCovs.append(np.cov(np.asmatrix(XXX[i]).T))

b=frame['label'].value_counts()
p=[]
S=[]
m=[[],[],[],[],[]]

for i in range(0,5):
    if(i not in b):
        p.append(0)
    else:
        p.append(b[i]*1.0/300)
    S.append(initialCovs[i])
    m[i]=V[i]

# print p
# print S
# print m

XX=np.asmatrix(X)

h=np.zeros((300,5))

#EM ALGORITHM:
print "Iterations:"
for i in range(0,100):
    print i+1
# E step
    for k in range(0,300):
        tempSum=0
        for j in range(0,5):
            xx=np.matrix([X[k]-m[j]])
            mat=xx.dot(np.linalg.inv(S[j])).dot(xx.T)
            mat=mat*(-.5)

            h[k][j]=p[j]*pow(np.linalg.det(S[j]),-0.5)*pow(e,mat[0])
            tempSum+=h[k][j]
        h[k]/=tempSum

#M step
    m=h.T.dot(XX)
    tempHsum=np.sum(h, axis=0)

    m=m/tempHsum[:,None]
    m= np.asarray(m)

    S=[]

    for j in range(0, 5):
        tempSum=0
        for k in range(0,300):
            xx=np.matrix([X[k]-m[j]])
            mat=xx.T.dot(xx)
            res=mat*h[k][j]
            tempSum+=res
        tempSum/=tempHsum[j]
        S.append(tempSum)

print "Mean Vectors:"
print m

#PLOTTING DENSITIES:
aa, bb = np.mgrid[-6:6:.05, -6:6:.05]
pos = np.empty(aa.shape + (2,))
pos[:, :, 0] = aa; pos[:, :, 1] = bb

colors = ['c', 'm', 'y', 'k']
for i in range(0,5):
    rv = multivariate_normal(mean[i],cov[i])
    ax2.contour(aa, bb, rv.pdf(pos),linestyles='dashed',levels = [0.05],colors='k')
    rv = multivariate_normal(m[i], S[i])
    ax2.contour(aa, bb, rv.pdf(pos),colors='k',levels = [0.05])

labels = []
for indexer, row in frame.iterrows():
    minDistance = 999
    bestLabel = -1
    for j in range(0, 5):
        tempDistance = sqrt((row[0] - m[j][0]) * (row[0] - m[j][0]) + (row[1] - m[j][1]) * (row[1] - m[j][1]))
        if (tempDistance < minDistance):
            minDistance = tempDistance
            bestLabel = j
    labels.append(bestLabel)


colors = {0:'purple', 1:'red', 2:'orange', 3:'blue',4:'green'}
#print frame
frame['label']=labels
ax2.scatter(frame['x'], frame['y'], c=frame['label'].apply(lambda x: colors[x]),marker='.')

a,b=np.asmatrix(X).T
ax1.plot(a, b,'k.')

ax1.set_xlabel('x1')
ax1.set_ylabel('x2')

ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
plt.show()