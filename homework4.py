import matplotlib.pyplot as plt
from math import sqrt
from math import e, pow,floor,pi
import pandas as pd
import numpy as np


from sklearn.neighbors.kde import KernelDensity

#READING DATA AND INITILAZING

data = pd.read_csv('hw04_data_set.csv')
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
data=data.reindex(np.random.permutation(data.index))
N=133
N_train=100
N_test=33


#REGRESSOGRAM

ax1.plot(data['x'][0:N_train],data['y'][0:N_train],'b.' ,label='training')
ax1.plot(data['x'][N_train:N],data['y'][N_train:N],'r.', label='test')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Regressogram h=3')
ax1.legend()

k=3
X_min=data['x'][0:N_train].min()
X_max=data['x'][0:N_train].max()
bin_number=floor(X_max/k)
x_bins=[]
y_bins=[]

for i in range(0,int(bin_number)+1):
    x_bins.append(0)
    y_bins.append(0)

for index, i in data[0:N_train].iterrows():
    x_bins[int(i['x']/k)]=x_bins[int(i['x']/k)]+1
    y_bins[int(i['x']/k)]=y_bins[int(i['x']/k)]+i['y']

for i in range(0,len(y_bins)):
    if(x_bins[i]!=0):
        y_bins[i]=y_bins[i]/x_bins[i]

for i in range(0,int(bin_number)+1):
    ax1.plot([i*k,i*k+k],[y_bins[i],y_bins[i]],'k')
    if(i+1<len(y_bins)):
        ax1.plot([i*k+k,i*k+k],[y_bins[i],y_bins[i+1]],'k')

error1=0
for i in range(N_train,N):
    error1=error1+pow(y_bins[int(data['x'][i]/k)]-data['y'][i],2)

error1=error1/N_test
error1=sqrt(error1)

print "Regressogram => RMSE is " + str(error1) +" when h is "+str(k)

#RUNNING MEAN SMOOTHER

ax2.plot(data['x'][0:N_train],data['y'][0:N_train],'b.' ,label='training')
ax2.plot(data['x'][N_train:N],data['y'][N_train:N],'r.', label='test')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Running Mean Smoother h=3')
ax2.legend()

train_data_sorted= data[0:N_train].sort_values(by=['x'])

meanEstimate=[]
a= train_data_sorted['x'].min() - 1.499
b= train_data_sorted['x'].max() + 1.499
meanRange=np.arange(a,b,0.1)

for i in meanRange:
    sum=0
    counter=0
    for index, j in train_data_sorted.iterrows():
        if(j['x']<i+1.5 and i-1.5<=j['x']):
            sum=sum+j['y']
            counter=counter+1
    if(counter==0):
        meanEstimate.append(meanEstimate[-1])
    else:
        meanEstimate.append(sum/counter)



map={}
for i in range(len(meanRange)):
    map[round(meanRange[i],1)]=meanEstimate[i]

error2=0
for i in range(N_train,N):
    error2=error2+pow(map[round(data['x'][i],1)]-data['y'][i],2)

error2=error2/N_test
error2=sqrt(error2)
print "Running Mean Smoother => RMSE is " + str(error2) +" when h is "+str(k)
ax2.plot(meanRange,meanEstimate,'k-')

#KERNEL SMOOTHER

ax3.plot(data['x'][0:N_train],data['y'][0:N_train],'b.' ,label='training')
ax3.plot(data['x'][N_train:N],data['y'][N_train:N],'r.', label='test')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Kernel Smoother h=1')
ax3.legend()

def kernel(x):
    return (1.0/sqrt(2.0*pi))*pow(e,(-x*x/2.0))

kernelEstimate=[]
kernelRange=np.arange(0,60,0.05)

kernelWidth=1
for i in kernelRange:
    sum=0
    generalsum=0
    for index,j in train_data_sorted.iterrows():
        sum=sum+(kernel((i-j['x'])/kernelWidth)*j['y'])
        generalsum=generalsum+kernel((i-j['x'])/kernelWidth)
    kernelEstimate.append(sum/generalsum)

X=[[],[]]
for index,j in train_data_sorted.iterrows():
    X[0].append(j['x'])
    X[1].append(j['y'])

kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(X)
map={}
for i in range(len(kernelRange)):
    map[round(kernelRange[i],1)]=kernelEstimate[i]

error3=0
for i in range(N_train,N):
    error3=error3+pow(map[round(data['x'][i],1)]-data['y'][i],2)

error3=error3/N_test
error3=sqrt(error3)
print "Kernel Smoother => RMSE is " + str(error3) +" when h is "+str(kernelWidth)
ax3.plot(kernelRange,kernelEstimate,'k-')

plt.show()