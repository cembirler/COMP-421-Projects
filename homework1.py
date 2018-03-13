import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv,det
from math import log

mean = [(0.0, 1.5),(-2.5,-3.0),(2.5,-3.0)]
cov=[[[1.0,0.2],[0.2,3.2]],[[1.6,-0.8],[-0.8,1.0]],[[1.6,0.8],[0.8,1.0]]]
n=[100,100,100]

x, y= np.random.multivariate_normal(mean[0], cov[0], n[0]).T
z, t= np.random.multivariate_normal(mean[1], cov[1], n[1]).T
k, l= np.random.multivariate_normal(mean[2], cov[2], n[2]).T

plt.plot(x, y,'r.')
plt.plot(z, t,'g.')
plt.plot(k, l,'b.')

variables=[x,y,z,t,k,l]

sample_means=[['xmean','zmean','kmean'],
              ['ymean','tmean','lmean']] # array to store mean values

for i in range(0,6):
    temp_mean=0
    for j in range(0,n[i/2]):
        temp_mean=temp_mean+variables[i][j]
    temp_mean=temp_mean/n[i/2]
    sample_means[i%2][i/2]=temp_mean


print "Sample means:"
for i in range(0,2):
    print sample_means[i]
print" "




sample_covariances=[[['xx','xy'],
                     ['yx','yy']],

                    [['zz','zt'],
                     ['tz','tt']],

                    [['kk','kl'],
                     ['lk','ll']]]

for i in range(0,3):
    temp_sum1 = 0
    temp_sum2 = 0
    temp_sum3 = 0
    temp_sum4 = 0
    for m in range(0,n[i]):
        temp_sum1=temp_sum1+(variables[i*2][m]-sample_means[0][i])*(variables[i*2][m]-sample_means[0][i])
        temp_sum2 = temp_sum2 + (variables[i * 2][m] - sample_means[0][i])*(variables[i * 2+1][m] - sample_means[1][i])
        temp_sum3 = temp_sum3 + (variables[i * 2+1][m] - sample_means[1][i])*(variables[i * 2][m] - sample_means[0][i])
        temp_sum4 = temp_sum4 + (variables[i * 2+1][m] - sample_means[1][i])*(variables[i * 2+1][m] - sample_means[1][i])
    temp_sum1 = temp_sum1 / n[i]
    temp_sum2 = temp_sum2 / n[i]
    temp_sum3 = temp_sum3 / n[i]
    temp_sum4 = temp_sum4 / n[i]

    sample_covariances[i][0][0] = temp_sum1
    sample_covariances[i][0][1] = temp_sum2
    sample_covariances[i][1][0] = temp_sum3
    sample_covariances[i][1][1] = temp_sum4

print "Sample covariances:"
for i in range(0,3):
    print (i+1)
    print sample_covariances[i][0]
    print sample_covariances[i][1]

print" "

class_priors=['1','2','3']
N=0
for i in range(0,3):
    N=N+n[i]
for i in range(0,3):
    class_priors[i]=float(n[i])/float(N)

print "Class priors:"
print class_priors
print" "

W=['1','2','3']
w=['1','2','3']
w0=['1','2','3']
X=['1','2','3']

means=np.matrix(sample_means).T


for i in range(0,len(n)):
    W[i]=(-0.5)*inv(np.matrix(sample_covariances[i]))
    w[i]=inv(np.matrix(sample_covariances[i]))*means[i].T
    w0[i]=(-0.5)*means[i]*inv(np.matrix(sample_covariances[i]))*means[i].T\
          -(0.5)*log(det(np.matrix(sample_covariances[i])))+log(class_priors[i])

    def g(x,i):
      return np.matrix(x).T*W[i]*x+w[i].T*x+w0[i]



confusion_matrix=[[0,0,0],[0,0,0],[0,0,0]]

for i in range(0,3):
    for k in range(0, n[i]):
        a=[]
        for j in range(0,3):
            a.append(g([[variables[2*i][k]],[variables[2*i+1][k]]],j))
        s=np.argmax(a)
        if(s != i):
            if (i==0):
                plt.plot([variables[2*i][k]],[variables[2*i+1][k]], 'ro')
            if (i==1):
                plt.plot([variables[2*i][k]],[variables[2*i+1][k]], 'go')
            if (i==2):
                plt.plot([variables[2*i][k]],[variables[2*i+1][k]], 'bo')
        confusion_matrix[s][i]=confusion_matrix[s][i]+1


print "Confusion matrix:"
for i in range(0,3):
    print confusion_matrix[i]

xx=np.arange(-6.0,6.0,0.05)
yy=np.arange(-6.0,6.0,0.05)


def class_function(x):
    if (g(x,0)>g(x,1) and g(x,0)>g(x,2)):
        return 0
    if (g(x,1)>g(x,0) and g(x,1)>g(x,2)):
        return 1
    if (g(x,2)>g(x,0) and g(x,2)>g(x,1)):
        return 2


Z=[]
for a in yy:
    for b in xx:
        Z.append(class_function([[a],[b]]))

Z=np.array(Z)
Z=Z.reshape((xx.size,yy.size))
Z=Z.T

X, Y = np.meshgrid(xx, yy)

levels=[-0.5,0.5,1.5,2.5]
cp = plt.contour(X, Y, Z, colors='k', linewidths=0.5)
cp = plt.contourf(X, Y, Z, levels=levels, colors=['#ff9898','#98ffa1','#98cfff'])

plt.show()