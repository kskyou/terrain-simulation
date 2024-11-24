
import math
import numpy as np
import pyvista as pv
from hkb_diamondsquare import DiamondSquare as DS

import taichi as ti
import taichi.math as tm

N = 1024 # Size of image
num_frames = 5000 # Number of frames

h_scale = 1. 
u_scale = 0.4 * h_scale / math.sqrt(N) 
p = 4
dt = 0.003

u_host = DS.diamond_square(shape=(N,N), min_height=0., max_height=u_scale, roughness=0.4) 
h_host = DS.diamond_square(shape=(N,N), min_height=0., max_height=h_scale, roughness=0.6)

ti.init()
u = ti.field(dtype=ti.f32)
h = ti.field(dtype=ti.f32)
h_new = ti.field(dtype=ti.f32)
a = ti.field(dtype=ti.f32)
a_new = ti.field(dtype=ti.f32)
Z = ti.field(dtype=ti.f32)
s = ti.field(dtype=ti.f32)

# Not sure how much this will help...
ti.root.dense(ti.ij, (N // 8, N // 8)).dense(ti.ij, (8, 8)).place(u)
ti.root.dense(ti.ij, (N // 8, N // 8)).dense(ti.ij, (8, 8)).place(h)
ti.root.dense(ti.ij, (N // 8, N // 8)).dense(ti.ij, (8, 8)).place(h_new)
ti.root.dense(ti.ij, (N // 8, N // 8)).dense(ti.ij, (8, 8)).place(a)
ti.root.dense(ti.ij, (N // 8, N // 8)).dense(ti.ij, (8, 8)).place(a_new)
ti.root.dense(ti.ij, (N // 8, N // 8)).dense(ti.ij, (8, 8)).place(Z)
ti.root.dense(ti.ij, (N // 8, N // 8)).dense(ti.ij, (8, 8)).place(s)

u.from_numpy(u_host)
h.from_numpy(h_host)

@ti.func
def slope(h,i1,j1,i2,j2):
    return (h[i2,j2] - h[i1,j1]) / (tm.sqrt((i2 - i1) ** 2 + (j2 - j1) ** 2))

@ti.func
def loop1(Z,s,h,i,j,x,y):
    if x >= 0 and x < N and y >= 0 and y < N and h[x,y] < h[i,j]:
        Z[i,j] += slope(h, x, y, i, j) ** p
        s[i,j] = max(s[i,j], slope(h, x, y, i, j))

@ti.func
def loop2(Z,a,a_new,h,i,j,x,y):
    if x >= 0 and x < N and y >= 0 and y < N and h[x,y] > h[i,j]:
        a_new[i,j] += (slope(h, i, j, x, y) ** p) / Z[x,y] * a[x,y]

@ti.kernel
def compute_z_s(h : ti.template(), Z : ti.template(), s : ti.template()):
    for i,j in h:
        Z[i,j] = 0.
        s[i,j] = 0.
        loop1(Z,s,h,i,j,i-1,j-1)
        loop1(Z,s,h,i,j,i-1,j)
        loop1(Z,s,h,i,j,i-1,j+1)
        loop1(Z,s,h,i,j,i,j-1)
        loop1(Z,s,h,i,j,i,j)
        loop1(Z,s,h,i,j,i,j+1)
        loop1(Z,s,h,i,j,i+1,j-1)
        loop1(Z,s,h,i,j,i+1,j)
        loop1(Z,s,h,i,j,i+1,j+1)
            
@ti.kernel
def update_a_h(h : ti.template(), Z : ti.template(), s : ti.template(), h_new : ti.template(), a : ti.template(), a_new : ti.template()):
    for i,j in h:
        a_new[i,j] = 1.
        loop2(Z,a,a_new,h,i,j,i-1,j-1)
        loop2(Z,a,a_new,h,i,j,i-1,j)
        loop2(Z,a,a_new,h,i,j,i-1,j+1)
        loop2(Z,a,a_new,h,i,j,i,j-1)
        loop2(Z,a,a_new,h,i,j,i,j)
        loop2(Z,a,a_new,h,i,j,i,j+1)
        loop2(Z,a,a_new,h,i,j,i+1,j-1)
        loop2(Z,a,a_new,h,i,j,i+1,j)
        loop2(Z,a,a_new,h,i,j,i+1,j+1)

        h_new[i,j] = max(h[i,j] + dt * (u[i,j] - tm.sqrt(a_new[i,j]) * s[i,j]), 0.0)

gui = ti.GUI("Terrain simulation", res=(N,N))

for t in range(num_frames):

    if t % 100 == 0:
        print("Frame %d" % t)

    compute_z_s(h, Z, s)

    update_a_h(h, Z, s, h_new, a, a_new)
    a, a_new = a_new, a
    h, h_new = h_new, h

    gui.set_image(h)
    gui.show()
    
h_host = h.to_numpy()
a_host = a.to_numpy()
s_host = s.to_numpy()
print("a min %f average %f max %f" % (np.min(a_host), np.mean(a_host), np.max(a_host)))
print("average lost %f" % np.mean(np.sqrt(a_host) * s_host))
print("h min %f max %f" % (np.min(h_host), np.max(h_host)))

xs = np.linspace(0, N, N)
ys = np.linspace(0, N, N)
xs, ys = np.meshgrid(xs, ys)
grid = pv.StructuredGrid(xs, ys, (h_host / h_scale * N * 0.2))
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars=grid.points[:, -1], cmap='gist_earth')
plotter.show_grid()
plotter.show()

