import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv,det
from math import log
from random import uniform
from math import e,pow

mean = [(0.0, 1.5),(-2.5,-3.0),(2.5,-3.0)]
cov=[[[1.0,0.2],[0.2,3.2]],[[1.6,-0.8],[-0.8,1.0]],[[1.6,0.8],[0.8,1.0]]]
n=[100,100,100]

X0= np.random.multivariate_normal(mean[0], cov[0], n[0]).T
X1= np.random.multivariate_normal(mean[1], cov[1], n[1]).T
X2= np.random.multivariate_normal(mean[2], cov[2], n[2]).T

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(X0[0], X0[1],'r.')
ax1.plot(X1[0], X1[1],'g.')
ax1.plot(X2[0], X2[1],'b.')



X=[]

for i in X0.T:
    X.append(i)
for i in X1.T:
    X.append(i)
for i in X2.T:
    X.append(i)

min1=99
min2=99
max1=-99
max2=-99



X=np.array(X)
z = np.ones((300,1))
X=np.append(z,X,axis=1)

for a in X:
    if(a[1]>max1):
        max1=a[1]
    if (a[1] < min1):
        min1 = a[1]
    if (a[2] > max2):
        max2 = a[2]
    if (a[2] < min2):
        min2 = a[2]
y=[]

# print min1
# print max1
# print min2
# print max2

for i in range(0,300):
    y.append(i/100)
y=np.array(y)
y=y.T

W = 0.01 * np.random.randn(3, 3)


errors=[[],[]]

confusion_matrix=[[0,0,0],[0,0,0],[0,0,0]]

r=[]
for i in range(0,100):
    r.append([1,0,0])
for i in range(0,100):
    r.append([0,1,0])
for i in range(0,100):
    r.append([0,0,1])

r=np.matrix(r)

errors=[[],[]]
for a in range(0,1200):
    #print a
    deltaW=np.zeros((3,3))

    error=0

    for t in range(0,300):
        phi=[0,0,0]
        for i in range(0,3):
            for j in range(0,3):
                phi[i]=phi[i]+W[i,j]*X[t,j]
        y=[0,0,0]
        sum=0;

        for i in range(0,3):
            sum=sum+pow(e,phi[i])
        for i in range(0,3):
            y[i]=pow(e,phi[i])/sum
            #print y[i]
        for i in range(0,3):
            error=error+r[t,i]*log(y[i])

        for i in range(0,3):
            for j in range(0,3):
                #print r[t,i]
                #print y[i]
                #print X[t,j]

                deltaW[i,j]=deltaW[i,j]+(r[t,i]-y[i])*X[t,j]
    #print deltaW
    for i in range(0,3):
        for j in range(0,3):
            W[i,j]=W[i,j]+0.001*deltaW[i,j]
    #print -error

    error=-error
    errors[0].append(a)
    errors[1].append(error)

ax2.plot(errors[0],errors[1],'k-')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Error')

xx=np.arange(min2-0.2,max2+0.2,0.05)

yy=np.arange(min1-0.2,max1+0.2,0.05)


def classifier(x):
    ans=np.dot(W, x)
    return np.argmax(ans)


for i in range(0,300):

    confusion_matrix[classifier(X[i])][i/100]=confusion_matrix[classifier(X[i])][i/100]+1
    if( i/100 !=classifier(X[i])):
        if(i/100==0):
            ax1.plot(X[i][1], X[i][2], 'ro')
        if (i / 100 == 1):
            ax1.plot(X[i][1], X[i][2], 'go')
        if (i / 100 == 2):
            ax1.plot(X[i][1], X[i][2], 'bo')




print "W:"
print W[1]
print W[2]
print
print "w0:"
print W[0]
print
print "Confusion Matrix:"
print
print "            y-truth"
print "y-predicted 1  2  3"
print "         1 "+str(confusion_matrix[0][0])+"  "+str(confusion_matrix[0][1])+"  "+str(confusion_matrix[0][2])
print "         2 "+str(confusion_matrix[1][0])+"  "+str(confusion_matrix[1][1])+"  "+str(confusion_matrix[1][2])
print "         3 "+str(confusion_matrix[2][0])+"  "+str(confusion_matrix[2][1])+"  "+str(confusion_matrix[2][2])



Z=[]
for a in yy:
    for b in xx:
        Z.append(classifier([1,a,b]))

Z=np.array(Z)
Z=Z.reshape((yy.size,xx.size))
Z=Z.T

X, Y = np.meshgrid(yy, xx)

levels=[-0.5,0.5,1.5,2.5]
cp = ax1.contour(X, Y, Z, colors='k')
cp = ax1.contourf(X, Y, Z, levels=levels, colors=['#ff9898','#98ffa1','#98cfff'])



plt.show()