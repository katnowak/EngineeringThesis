# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:13:28 2020

@author: Kasia
"""

import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
#   3D plot of the
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot


# Load the STL files and add the vectors to the plot
your_mesh1 = mesh.Mesh.from_file('p4bkr.stl')
print(your_mesh1.v0[:,0])
# Load the STL files and add the vectors to the plot
your_mesh2 = mesh.Mesh.from_file('s4bkr.stl')
print(your_mesh2.v0[:,0])

%matplotlib qt

def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = np.sqrt(t)

    return radius, C[0], C[1], C[2]


#nowa metoda
def sphereFitNew(vert):
    from scipy import optimize
    from scipy.spatial.distance import cdist

    def f(c):
        centr_arr = [c]
        dist_arr = cdist(centr_arr, vert, 'euclidean').flatten()
        return dist_arr - dist_arr.mean()
        
    center_estimate = np.mean(vert, axis = 0)
    center, ier = optimize.leastsq(f, center_estimate)
    radius = cdist([center], vert, 'euclidean').flatten().mean()
    
    return radius, center[0], center[1], center[2]




correctX1 = your_mesh1.v0[:,0]
correctY1 = your_mesh1.v0[:,1]
correctZ1 = your_mesh1.v0[:,2]

#r1, x01, y01, z01 = sphereFit(correctX1,correctY1,correctZ1)
r1, x01, y01, z01 = sphereFitNew(your_mesh1.v0)


u1, v1 = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x1=np.cos(u1)*np.sin(v1)*r1
y1=np.sin(u1)*np.sin(v1)*r1
z1=np.cos(v1)*r1
x1 = x1 + x01
y1 = y1 + y01
z1 = z1 + z01

correctX2 = your_mesh2.v0[:,0]
correctY2 = your_mesh2.v0[:,1]
correctZ2 = your_mesh2.v0[:,2]

#r2, x02, y02, z02 = sphereFit(correctX2,correctY2,correctZ2)
r2, x02, y02, z02 = sphereFitNew(your_mesh2.v0)

u2, v2 = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x2=np.cos(u2)*np.sin(v2)*r2
y2=np.sin(u2)*np.sin(v2)*r2
z2=np.cos(v2)*r2
x2 = x2 + x02
y2 = y2 + y02
z2 = z2 + z02

#   3D plot of Sphere
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(correctX1, correctY1, correctZ1, zdir='z', s=40, c='magenta',rasterized=True, visible=False)
ax.plot_wireframe(x1, y1, z1, color="lime")

ax.scatter(correctX2, correctY2, correctZ2, zdir='z2', s=40, c='cyan',rasterized=True, visible=False)
ax.plot_wireframe(x2, y2, z2, color="black")
ax.set_aspect('auto')
ax.set_xlim3d(-45,-17)
ax.set_ylim3d(5,35)
ax.set_zlim3d(-65,-42)
ax.set_xlabel('$x$ [mm]',fontsize=12)
ax.set_ylabel('\n$y$ [mm]',fontsize=12)
zlabel = ax.set_zlabel('\n$z$ [mm]',fontsize=12)


#plt.savefig("p4b_s3b_met2.png", dpi=500, facecolor='w',optimize=True, edgecolor='w', orientation='portrait', papertype=None, format=None,
       # transparent=False, bbox_inches='tight', pad_inches=0.1,
       # frameon=None, metadata=None)

plt.show()


#######################
## sfera 1 - your_mesh1
import math

 

print(" - STATYSTYKA - pierwsza sfera - your_mesh1")

 

odl_lista = []

 

liczba_wierzcholkow = your_mesh1.v0.shape[0]
 

 

for i in range(liczba_wierzcholkow):
    aktualny = your_mesh1.v0[i, :]
    
    odl = math.sqrt((aktualny[0] - x01) * (aktualny[0] - x01) + (aktualny[1] - y01) * (aktualny[1] - y01) + (aktualny[2] - z01) * (aktualny[2] - z01))
    odl_lista.append(odl)
    
    #print("[", i, "]", "odl:", odl, " roznica:", odl - r1)
    
odl_macierz = np.array(odl_lista)

 

odl_pkt_od_sfery = odl_macierz - r1
print("srednia odl od srodka:", np.mean(odl_macierz), " odch. stand.:", np.std(odl_macierz))
print("srednia odl od sfery:", np.mean(odl_pkt_od_sfery), " srednia odl_po wart.bezw.:", np.mean(np.abs(odl_pkt_od_sfery))) 
print("^odch.1:", np.std(odl_pkt_od_sfery), " ^odch.2:", np.std(np.abs(odl_pkt_od_sfery))) 

 

#######################
## sfera 2 - your_mesh2
print(" - STATYSTYKA - druga sfera - your_mesh2")

 

odl_lista = []

 

liczba_wierzcholkow = your_mesh2.v0.shape[0]
 

 

for i in range(liczba_wierzcholkow):
    aktualny = your_mesh2.v0[i, :]
    
    odl = math.sqrt((aktualny[0] - x02) * (aktualny[0] - x02) + (aktualny[1] - y02) * (aktualny[1] - y02) + (aktualny[2] - z02) * (aktualny[2] - z02))
    odl_lista.append(odl)
    
    #print("[", i, "]", "odl:", odl, " roznica:", odl - r2)
    
odl_macierz = np.array(odl_lista)

 

odl_pkt_od_sfery = odl_macierz - r2
print("srednia odl od srodka:", np.mean(odl_macierz), " odch. stand.:", np.std(odl_macierz))
print("srednia odl od sfery:", np.mean(odl_pkt_od_sfery), " srednia odl_po wart.bezw.:", np.mean(np.abs(odl_pkt_od_sfery))) 
print("^odch.1:", np.std(odl_pkt_od_sfery), " ^odch.2:", np.std(np.abs(odl_pkt_od_sfery))) 


print('r piszczel= ', r1)

print ('r skok= ',r2)
