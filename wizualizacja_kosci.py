# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:33:39 2017

@author: Adam Ciszkiewicz
"""
import numpy as np
from UtilityLib import mesh_reader
from mayavi import mlab

#metody
#algebraiczna
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

#optymalizacyjna
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

############################################################
######## ladowanie siatek

### ponizej trzeba uzupelnic nazwy plikow stl
### pliki stl musza sie znajdowac w folderze z programem

## wycinek pierwszy
vert, tria, normals = mesh_reader('p3a.stl')
## wycinek drugi
vert_c, tria_c, normals_c = mesh_reader('s4akr.stl')

## kosc pierwsza
vertf, triaf, normalsf = mesh_reader('piszczelred.stl')
## kosc druga
vertf2, triaf2, normalsf2 = mesh_reader('skokowared.stl')


######### fitowanie
#r, x0, y0, z0 = sphereFitNew(vert)
#r1, x01, y01, z01 = sphereFitNew(vert_c)

r, x0, y0, z0 = sphereFit(vert[:, 0], vert[:, 1], vert[:, 2])
r1, x01, y01, z01 = sphereFit(vert_c[:, 0], vert_c[:, 1], vert_c[:, 2])


######## plotowanie
### utworzenie figury
mlab.figure(bgcolor=(1., 1., 1.))

### kosci
## kosc pierwsza
mlab.triangular_mesh(vertf[:,0], 
                     vertf[:,1], 
                     vertf[:,2], 
                     triaf, color=(0.8, 0.8, 0.8),
                     opacity = .7)

## kosc druga
mlab.triangular_mesh(vertf2[:,0], 
                     vertf2[:,1], 
                     vertf2[:,2], 
                     triaf2, color=(0.8, 0.8, 0.8),
                     opacity = .7)

#### wycinki do par stykowych
## pierwszy
mlab.triangular_mesh(vert[:,0], 
                      vert[:,1], 
                      vert[:,2], 
                      tria, color=(0.9, 0.1, 0.8),
                      opacity = 1.0)

mlab.points3d(vert[:,0],
              vert[:,1], 
              vert[:,2],
              color=(0., 0., 0.),
              scale_factor = 0.25)


## drugi
mlab.triangular_mesh(vert_c[:,0],
                      vert_c[:,1], 
                      vert_c[:,2], 
                      tria_c, color=(0.4, 0.7, 1.))

mlab.points3d(vert_c[:,0],
              vert_c[:,1], 
              vert_c[:,2],
              color=(0., 0., 0.),
              scale_factor = 0.25)

###########
### sfery
# obliczenia
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
x=np.cos(u)*np.sin(v)*r
y=np.sin(u)*np.sin(v)*r
z=np.cos(v)*r
x = x + x0
y = y + y0
z = z + z0

u1, v1 = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
x1=np.cos(u1)*np.sin(v1)*r1
y1=np.sin(u1)*np.sin(v1)*r1
z1=np.cos(v1)*r1
x1 = x1 + x01
y1 = y1 + y01
z1 = z1 + z01


# rysowanie sfer
mlab.mesh(x, y, z, representation='surface', 
          color=(0.5, 1., 0.2),
          opacity = 0.6)


mlab.mesh(x1, y1, z1, representation='surface', 
          color=(0., 0., 0.1),
          opacity = 0.4)

# rysowanie srodkow sfer
mlab.points3d(x0, y0, z0, 
              color=(1., 0., 0.),
              opacity = 1.)

mlab.points3d(x01, y01, z01,          
              color=(0., 0., 1.),
              opacity = 1.)

# wyswietlanie figury
mlab.show()