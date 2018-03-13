import numpy as np
import matplotlib.pyplot as plt
from math import log
from math import e, pow

mean = [(2,2),
        (-4,-4),
        (-2,2),
        (4,-4),
        (-2,-2),
        (4,4),
        (2,-2),
        (-4,4)]

cov=[[[0.8,-0.6],[-0.6,0.8]],
     [[0.4,0],[0,0.4]],
     [[0.8,0.6],[0.6,0.8]],
     [[0.4,0],[0,0.4]],
     [[0.8,-0.6],[-0.6,0.8]],
     [[0.4, 0], [0, 0.4]],
     [[0.8, 0.6], [0.6, 0.8]],
     [[0.4, 0], [0, 0.4]]]


n=50
X=[[],[],[],[]] #feature0,feature1,feature2,class


for i in range(0,8):
    d1,d2=np.random.multivariate_normal(mean[i], cov[i], n).T
    for j in range(0,50):
        X[0].append(1)
        X[1].append(d1[j])
        X[2].append(d2[j])
        X[3].append(i/2)



f, (ax1, ax2) = plt.subplots(1, 2)

colors=['r.','g.','b.','y.']
for i in range(0,400):
    ax1.plot(X[1][i],X[2][i],colors[X[3][i]])

X=np.array(X).T

min1=99
min2=99
max1=-99
max2=-99

for a in X:
    if(a[1]>max1):
        max1=a[1]
    if (a[1] < min1):
        min1 = a[1]
    if (a[2] > max2):
        max2 = a[2]
    if (a[2] < min2):
        min2 = a[2]


confusion_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]


#initilaztion of weights:
V = 0.01 * np.random.randn(4, 21) #class,hidden number of outputs,
W = 0.01 * np.random.randn(21, 3) #hidden number of outpust, features

def sigmoid(x):
    return 1/(1+pow(e,-x))

def r(X,i):
    ans=[]
    if (X[3] == 0):
        ans = [1, 0, 0, 0]
    if (X[3] == 1):
        ans = [0, 1, 0, 0]
    if (X[3] == 2):
        ans = [0, 0, 1, 0]
    if (X[3] == 3):
        ans = [0, 0, 0, 1]
    return ans[i]

print "Iteration   Error"
errors=[[],[]]
for a in range(0,200):
    np.random.shuffle(X)
    error=0
    for t in range(0,400):
        z=np.zeros((21,1))
        z[0]=1
        for h in range(1,21):
            z[h]=sigmoid(np.dot(W[h],[X[t,0],X[t,1],X[t,2]]))

        y=np.zeros((4,1))

        phi=np.zeros((4,1))

        for i in range(0,4):
            sumphi=0
            for h in range(0,21):
                sumphi=sumphi+V[i,h]*z[h]
            phi[i]=sumphi

        sumy=0
        for i in range(0,4):
            sumy=sumy+pow(e,phi[i])
        for i in range(0,4):
            y[i]=pow(e,phi[i])/sumy
            error = error + r(X[t], i) * log(y[i])

        deltav=np.zeros((4,21))
        deltaw=np.zeros((21,3))



        for i in range(0,4):
            for h in range(0,21):
                deltav[i,h]=0.1*(r(X[t],i)-y[i])*z[h]

        for h in range(0,21):
            for j in range(0,3):
                sum=0
                for i in range(0,4):
                    sum=sum+(r(X[t],i)-y[i])*V[i,h]
                deltaw[h,j]=0.1*sum*z[h]*(1-z[h])*X[t,j]

        for i in range(0,4):
            for h in range(0,21):
                V[i,h]=V[i,h]+deltav[i,h]

        for h in range(0,21):
            for j in range(0,3):
                W[h,j]=W[h,j]+deltaw[h,j]

    error=-error
    print str(a+1)+ "           "+str(error)
    errors[0].append(a)
    errors[1].append(error)
    if(a>0 and a<200):
        if(abs(errors[1][a]-errors[1][a-1])<0.001):
            break


ax2.plot(errors[0],errors[1],'k-')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Error')


def classifier(x):
    z1=np.zeros((21,1))
    for i in range(0,21):
        z1[i]=sigmoid(np.dot(W[i],[[x[0]],[x[1]],[x[2]]]))

    y=np.zeros((4,1))
    sum=0
    for i in range(0,4):
        sum=sum+pow(e,np.dot(V[i],z1))
    for i in range(0,4):
        y[i]=pow(e,np.dot(V[i],z1))/sum

    return np.argmax(y)


for i in range(0,400):
    confusion_matrix[classifier(X[i])][int(X[i,3])]=confusion_matrix[classifier(X[i])][int(X[i,3])]+1
    if( int(X[i,3]) !=classifier(X[i])):
        if(X[i,3]==0):
            ax1.plot(X[i][1], X[i][2], 'ro')
        if (X[i,3] == 1):
            ax1.plot(X[i][1], X[i][2], 'go')
        if (X[i,3] == 2):
            ax1.plot(X[i][1], X[i][2], 'bo')
        if (X[i,3] == 3):
            ax1.plot(X[i][1], X[i][2], 'yo')
print
print "Confusion Matrix:"
print
print "             y-truth"
print "y-predicted  1  2  3 4"
print "         1  "+str(confusion_matrix[0][0])+"  "+str(confusion_matrix[0][1])+"  "+str(confusion_matrix[0][2])+"  "+str(confusion_matrix[0][3])
print "         2  "+str(confusion_matrix[1][0])+"  "+str(confusion_matrix[1][1])+"  "+str(confusion_matrix[1][2])+"  "+str(confusion_matrix[1][3])
print "         3  "+str(confusion_matrix[2][0])+"  "+str(confusion_matrix[2][1])+"  "+str(confusion_matrix[2][2])+"  "+str(confusion_matrix[2][3])
print "         4  "+str(confusion_matrix[3][0])+"  "+str(confusion_matrix[3][1])+"  "+str(confusion_matrix[3][2])+"  "+str(confusion_matrix[3][3])

Z=[]
xx=np.arange(min2-0.2,max2+0.2,0.05)
yy=np.arange(min1-0.2,max1+0.2,0.05)
for a in yy:
    for b in xx:
        Z.append(classifier([1,a,b]))

Z=np.array(Z)
Z=Z.reshape((yy.size,xx.size))
Z=Z.T

X, Y = np.meshgrid(yy, xx)

levels=[-0.5,0.5,1.5,2.5,3.5]
cp = ax1.contour(X, Y, Z, colors='k', linewidths=0.5)
cp = ax1.contourf(X, Y, Z, levels=levels, colors=['#ff9898','#98ffa1','#98cfff','#fdf5c9'])



plt.show()