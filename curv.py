import cv2 
import math
import pylab
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as nd
import numpy as np

from scipy.ndimage.measurements import label
from sklearn.cluster import KMeans



def findCurvature(X, Y, Z):

	rows = X.shape[0]
	cols = X.shape[1]

	(X2, X1), (Y2, Y1), (Z2, Z1) = np.gradient(X), np.gradient(Y), np.gradient(Z)
	(X12, X11), (Y12, Y11), (Z12, Z11), (X22, X12), (Y22, Y12), (Z22, Z12) = np.gradient(X1), np.gradient(Y1), np.gradient(Z1), np.gradient(X2), np.gradient(Y2), np.gradient(Z2)   

	linr = rows*cols

	X1, Y1, Z1, X2, Y2, Z2  = np.reshape(X1, linr), np.reshape(Y1, linr), np.reshape(Z1, linr), np.reshape(X2, linr), np.reshape(Y2, linr), np.reshape(Z2, linr)
	X11, Y11, Z11, X12, Y12, Z12, X22, Y22, Z22 = np.reshape(X11, linr), np.reshape(Y11, linr), np.reshape(Z11, linr), np.reshape(X12, linr), np.reshape(Y12, linr), np.reshape(Z12, linr), np.reshape(X22, linr), np.reshape(Y22, linr), np.reshape(Z22, linr)

	X1, X2 = np.c_[X1, Y1, Z1], np.c_[X2, Y2, Z2]
	X11, X12 = np.c_[X11, Y11, Z11], np.c_[X12, Y12, Z12]
	X22 = np.c_[X22, Y22, Z22]

#Calculation of first fundamental co-efficients	
	E, F, G = (X1 * X1).sum(axis=1), (X1 * X2).sum(axis=1), (X2 * X2).sum(axis=1)

	m = np.cross(X1, X2, axisa=1, axisb=1) 
	p = np.sqrt(np.einsum('ij,ij->i', m, m)) 
	n = m/np.c_[p, p, p]

#Calculation of second fundamental co-efficients	
	L, M, N = (X11 * n).sum(axis=1), (X12 * n).sum(axis=1), (X22 * n).sum(axis=1)

#Calculation of gaussian curvature
	K = ((L*N)-M**2)/((E*G)-F**2)

#Calculation of mean curvature
	H = (((E*N) + (G*L) - (2*F*M))/(((E*G) - F**2)))/2

	K = np.reshape(K, linr)
	H = np.reshape(H, linr)

#% Principle Curvatures
	P1 = H + np.sqrt(H**2 - K)
	P2 = H - np.sqrt(H**2 - K)

	Principle = [P1, P2]
	return Principle, K, H



image1 = cv2.imread('RGBD_dataset/2.png')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

x = np.zeros(image1.shape)
y = np.zeros(image1.shape)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        x[i][j] = j

for i in range(y.shape[0]):
    for j in range(y.shape[1]):
        y[i][j] = i

z = image1.copy()

print("Showing the disparity image: ")

fig = plt.figure()
fig.suptitle('Range Image: ', fontsize = 20)
plt.imshow(image1)
plt.axis("off")
plt.show()

print("Visual representation of depth map: ")

fig = pylab.figure()
fig.suptitle('3D Representation (please rorate): ', fontsize = 20)
ax = Axes3D(fig)
ax.plot_surface(x,y,z)
pylab.show()


print("Calculating Topology based on principal curvature: ")
pC, gC, mC = findCurvature(x, y, z)

errTolerence = 0.8

peak = (pC[0] < -errTolerence) * (pC[1] < -errTolerence)
flat = (pC[0] < errTolerence) * (pC[0] > -errTolerence) * (pC[1] < errTolerence) * (pC[1] > -errTolerence)
pit = (pC[0] > errTolerence) * (pC[1] > errTolerence)

ridge1 = (pC[0] < errTolerence) * (pC[0] > -errTolerence) * (pC[1] < -errTolerence)
ridge2 = (pC[1] < errTolerence) * (pC[1] > -errTolerence) * (pC[0] < -errTolerence)
ridge = ridge1 + ridge2

saddle1 = (pC[0] < -errTolerence) * (pC[1] > errTolerence)
saddle2 = (pC[0] > errTolerence) * (pC[1] < -errTolerence)
saddle = saddle1 + saddle2

valley1 = (pC[0] < errTolerence) * (pC[0] > -errTolerence) * (pC[1] > errTolerence)
valley2 = (pC[1] < errTolerence) * (pC[1] > -errTolerence) * (pC[0] > errTolerence)
valley = valley1 + valley2



flatMap = image1.copy()
peakMap = image1.copy()
pitMap = image1.copy()
ridgeMap = image1.copy()
saddleMap = image1.copy()
valleyMap = image1.copy()

for i in range(flatMap.shape[0]):
    for j in range(flatMap.shape[1]):
        if(flat[(i * flatMap.shape[1]) + j]):
            flatMap[i][j] = 255
        else:
            flatMap[i][j] = 0

        if(peak[(i * peakMap.shape[1]) + j]):
            peakMap[i][j] = 255
        else:
            peakMap[i][j] = 0

        if(pit[(i * pitMap.shape[1]) + j]):
            pitMap[i][j] = 255
        else:
            pitMap[i][j] = 0

        if(ridge[(i * ridgeMap.shape[1]) + j]):
            ridgeMap[i][j] = 255
        else:
            ridgeMap[i][j] = 0

        if(saddle[(i * saddleMap.shape[1]) + j]):
            saddleMap[i][j] = 255
        else:
            saddleMap[i][j] = 0

        if(valley[(i * valleyMap.shape[1]) + j]):
            valleyMap[i][j] = 255
        else:
            valleyMap[i][j] = 0


print("Plotting Results: ")

fig = plt.figure()
fig.suptitle('Flat Map: ', fontsize = 20)
plt.imshow(flatMap)
plt.axis("off")
plt.show()

fig = plt.figure()
fig.suptitle('Peak Map: ', fontsize = 20)
plt.imshow(peakMap)
plt.axis("off")
plt.show()

fig = plt.figure()
fig.suptitle('Pit Map: ', fontsize = 20)
plt.imshow(pitMap)
plt.axis("off")
plt.show()

fig = plt.figure()
fig.suptitle('Ridge Map: ', fontsize = 20)
plt.imshow(ridgeMap)
plt.axis("off")
plt.show()

fig = plt.figure()
fig.suptitle('Saddle Map: ', fontsize = 20)
plt.imshow(saddleMap)
plt.axis("off")
plt.show()

fig = plt.figure()
fig.suptitle('Valley Map: ', fontsize = 20)
plt.imshow(valleyMap)
plt.axis("off")
plt.show()



print("Calculating Topology based on mean and Gaussian curvatures: ")

peak = (mC < -errTolerence) * (gC < -errTolerence)
flat = (mC < errTolerence) * (mC > -errTolerence) * (gC < errTolerence) * (gC > -errTolerence)
pit = (mC > errTolerence) * (gC > errTolerence)

ridge1 = (mC < errTolerence) * (mC > -errTolerence) * (gC < -errTolerence)
ridge2 = (gC < errTolerence) * (gC > -errTolerence) * (mC < -errTolerence)
ridge = ridge1 + ridge2

saddle1 = (mC < -errTolerence) * (gC > errTolerence)
saddle2 = (mC > errTolerence) * (gC < -errTolerence)
saddle = saddle1 + saddle2

valley1 = (mC < errTolerence) * (mC > -errTolerence) * (gC > errTolerence)
valley2 = (gC < errTolerence) * (gC > -errTolerence) * (mC > errTolerence)
valley = valley1 + valley2

flatMap2 = image1.copy()
peakMap2 = image1.copy()
pitMap2 = image1.copy()
ridgeMap2 = image1.copy()
saddleMap2 = image1.copy()
valleyMap2 = image1.copy()

for i in range(flatMap2.shape[0]):
    for j in range(flatMap2.shape[1]):
        if(flat[(i * flatMap2.shape[1]) + j]):
            flatMap2[i][j] = 1
        else:
            flatMap2[i][j] = 0

        if(peak[(i * peakMap2.shape[1]) + j]):
            peakMap2[i][j] = 1
        else:
            peakMap2[i][j] = 0

        if(pit[(i * pitMap2.shape[1]) + j]):
            pitMap2[i][j] = 1
        else:
            pitMap2[i][j] = 0

        if(ridge[(i * ridgeMap2.shape[1]) + j]):
            ridgeMap2[i][j] = 1
        else:
            ridgeMap2[i][j] = 0

        if(saddle[(i * saddleMap2.shape[1]) + j]):
            saddleMap2[i][j] = 1
        else:
            saddleMap2[i][j] = 0

        if(valley[(i * valleyMap2.shape[1]) + j]):
            valleyMap2[i][j] = 1
        else:
            valleyMap2[i][j] = 0


print("Plotting Results: ")

fig = plt.figure()
fig.suptitle('Flat Map: ', fontsize = 20)
plt.imshow(flatMap2)
plt.axis("off")
plt.show()

fig = plt.figure()
fig.suptitle('Peak Map: ', fontsize = 20)
plt.imshow(peakMap2)
plt.axis("off")
plt.show()

fig = plt.figure()
fig.suptitle('Pit Map: ', fontsize = 20)
plt.imshow(pitMap2)
plt.axis("off")
plt.show()

fig = plt.figure()
fig.suptitle('Ridge Map: ', fontsize = 20)
plt.imshow(ridgeMap2)
plt.axis("off")
plt.show()

fig = plt.figure()
fig.suptitle('Saddle Map: ', fontsize = 20)
plt.imshow(saddleMap2)
plt.axis("off")
plt.show()

fig = plt.figure()
fig.suptitle('Valley Map: ', fontsize = 20)
plt.imshow(valleyMap2)
plt.axis("off")
plt.show()


res = 0.4
nT = 6
print("Calculating Neighborhood Plane Set for each pixel (Time: 2 mins)... ")

NPS = np.zeros([image1.shape[0], image1.shape[1], 9])

for i in range(1, image1.shape[0]-1):
    for j in range(1, image1.shape[1]-1):

        NPS[i][j][0] = 1
        NPS[i][j][1] = 1
        NPS[i][j][3] = 1
        NPS[i][j][4] = 1

        count = 0
        if(z[i][j+1] > (z[i][j] - res) and z[i][j+1] < (z[i][j] + res)):
            count += 1
        if(z[i][j-1] > (z[i][j] - res) and z[i][j-1] < (z[i][j] + res)):
            count += 1
        if(z[i+1][j] > (z[i][j] - res) and z[i+1][j] < (z[i][j] + res)):
            count += 1
        if(z[i-1][j] > (z[i][j] - res) and z[i-1][j] < (z[i][j] + res)):
            count += 1
        if(z[i-1][j+1] > (z[i][j] - res) and z[i-1][j+1] < (z[i][j] + res)):
            count += 1
        if(z[i-1][j-1] > (z[i][j] - res) and z[i-1][j-1] < (z[i][j] + res)):
            count += 1
        if(z[i+1][j+1] > (z[i][j] - res) and z[i+1][j+1] < (z[i][j] + res)):
            count += 1
        if(z[i+1][j-1] > (z[i][j] - res) and z[i+1][j-1] < (z[i][j] + res)):
            count += 1
        
        if(count >= nT):
            NPS[i][j][2] = 1


        count = 0
        if(z[i][j+1] > (z[i][j] + (res)) and z[i][j+1] < (z[i][j] + (2*res))):
            count += 1
        if(z[i][j-1] > (z[i][j] - (2*res)) and z[i][j-1] < (z[i][j] - res)):
            count += 1
        if(z[i+1][j] > (z[i][j] - res) and z[i+1][j] < (z[i][j] + res)):
            count += 1
        if(z[i-1][j] > (z[i][j] - res) and z[i-1][j] < (z[i][j] + res)):
            count += 1
        if(z[i-1][j+1] > (z[i][j] + (res)) and z[i-1][j+1] < (z[i][j] + (2*res))):
            count += 1
        if(z[i-1][j-1] > (z[i][j] - (2*res)) and z[i-1][j-1] < (z[i][j] - res)):
            count += 1
        if(z[i+1][j+1] > (z[i][j] + (res)) and z[i+1][j+1] < (z[i][j] + (2*res))):
            count += 1
        if(z[i+1][j-1] > (z[i][j] - (2*res)) and z[i+1][j-1] < (z[i][j] - res)):
            count += 1
        
        if(count >= nT):
            NPS[i][j][5] = 1

        
        count = 0
        if(z[i][j+1] > (z[i][j] - (2*res)) and z[i][j+1] < (z[i][j] - (res))):
            count += 1
        if(z[i][j-1] > (z[i][j] + (res)) and z[i][j-1] < (z[i][j] + (2*res))):
            count += 1
        if(z[i+1][j] > (z[i][j] - res) and z[i+1][j] < (z[i][j] + res)):
            count += 1
        if(z[i-1][j] > (z[i][j] - res) and z[i-1][j] < (z[i][j] + res)):
            count += 1
        if(z[i-1][j+1] > (z[i][j] - (2*res)) and z[i-1][j+1] < (z[i][j] - (res))):
            count += 1
        if(z[i-1][j-1] > (z[i][j] + (res)) and z[i-1][j-1] < (z[i][j] + (2*res))):
            count += 1
        if(z[i+1][j+1] > (z[i][j] - (2*res)) and z[i+1][j+1] < (z[i][j] - (res))):
            count += 1
        if(z[i+1][j-1] > (z[i][j] + (res)) and z[i+1][j-1] < (z[i][j] + (2*res))):
            count += 1
        
        if(count >= nT):
            NPS[i][j][6] = 1


        count = 0
        if(z[i][j+1] > (z[i][j] - res) and z[i][j+1] < (z[i][j] + res)):
            count += 1
        if(z[i][j-1] > (z[i][j] - res) and z[i][j-1] < (z[i][j] + res)):
            count += 1
        if(z[i+1][j] > (z[i][j] + (res)) and z[i+1][j] < (z[i][j] + (2*res))):
            count += 1
        if(z[i-1][j] > (z[i][j] - (2*res)) and z[i-1][j] < (z[i][j] - (res))):
            count += 1
        if(z[i-1][j+1] > (z[i][j] - (2*res)) and z[i-1][j+1] < (z[i][j] - (res))):
            count += 1
        if(z[i-1][j-1] > (z[i][j] - (2*res)) and z[i-1][j-1] < (z[i][j] - (res))):
            count += 1
        if(z[i+1][j+1] > (z[i][j] + (res)) and z[i+1][j+1] < (z[i][j] + (2*res))):
            count += 1
        if(z[i+1][j-1] > (z[i][j] + (res)) and z[i+1][j-1] < (z[i][j] + (2*res))):
            count += 1
        
        if(count >= nT):
            NPS[i][j][7] = 1

        count = 0
        if(z[i][j+1] > (z[i][j] - res) and z[i][j+1] < (z[i][j] + res)):
            count += 1
        if(z[i][j-1] > (z[i][j] - res) and z[i][j-1] < (z[i][j] + res)):
            count += 1
        if(z[i+1][j] > (z[i][j] - (2*res)) and z[i+1][j] < (z[i][j] - (res))):
            count += 1
        if(z[i-1][j] > (z[i][j] + (res)) and z[i-1][j] < (z[i][j] + (2*res))):
            count += 1
        if(z[i-1][j+1] > (z[i][j] + (res)) and z[i-1][j+1] < (z[i][j] + (2*res))):
            count += 1
        if(z[i-1][j-1] > (z[i][j] + (res)) and z[i-1][j-1] < (z[i][j] + (2*res))):
            count += 1
        if(z[i+1][j+1] > (z[i][j] - (2*res)) and z[i+1][j+1] < (z[i][j] - (res))):
            count += 1
        if(z[i+1][j-1] > (z[i][j] - (2*res)) and z[i+1][j-1] < (z[i][j] - (res))):
            count += 1
        
        if(count >= nT):
            NPS[i][j][8] = 1

    if(i%(image1.shape[0]/10)==0):
        print("Progress: ", i*100/image1.shape[0], " %")


print("Neighborhood Plane Set for each pixel calculated.")



print("Region growing based on Neighbourhood Plane Set to find connected components...")

NPS = NPS[:, :, 2]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
NPS = cv2.morphologyEx(NPS, cv2.MORPH_OPEN, kernel)

#NPS = np.reshape(NPS, [image1.shape[0]*image1.shape[1], 3])

labels, nCom = label(NPS)
outNPS = np.reshape(labels, image1.shape)

fig = plt.figure()
fig.suptitle('Segmentation based on NPS: ', fontsize = 20)
plt.imshow(outNPS)
plt.axis("off")
plt.show()

print("Region growing based on gaussian curvatures to find connected components...")

connectivity = np.ones((3, 3), dtype=np.int)
labels, nCom = label(flatMap2)

outNPS = np.reshape(labels, image1.shape)

fig = plt.figure()
fig.suptitle('Segmentation based on Gaussian Curvatures: ', fontsize = 20)
plt.imshow(outNPS)
plt.axis("off")
plt.show()


print("Region growing based on principal curvatures to find connected components...")

connectivity = np.ones((3, 3), dtype=np.int)
labels, nCom = label(flatMap)

outNPS = np.reshape(labels, image1.shape)

fig = plt.figure()
fig.suptitle('Segmentation based on Principal Curvatures: ', fontsize = 20)
plt.imshow(outNPS)
plt.axis("off")
plt.show()