
import math
import numpy as np
import skimage
import pyvista as pv
from hkb_diamondsquare import DiamondSquare as DS

def print_image(path, I, normalize=False):
    if normalize:
        I_scale = (I - np.min(I)) / (np.max(I) - np.min(I))
    else:
        I_scale = np.clip(I, 0, 1)
    skimage.io.imsave(path, (255 * I_scale).astype(np.uint8))

N = 256 # Size of image
num_frames = 100 # Number of frames

h_scale = 1. 
u_scale = 0.4 * h_scale / math.sqrt(N) 
p = 4
dt = 0.2

u = DS.diamond_square(shape=(N,N), min_height=0., max_height=u_scale, roughness=0.4) 
h = DS.diamond_square(shape=(N,N), min_height=0., max_height=h_scale, roughness=0.6)

print_image("out/uplift.png", u, normalize=True)

a = np.ones((N,N))
Z = np.ones((N,N))
s = np.ones((N,N))

def neighs(i,j):
    L = [[i-1,j-1],[i-1,j],[i-1,j+1], [i,j-1],[i,j],[i,j+1], [i+1,j-1],[i+1,j],[i+1,j+1]]
    return [(x,y) for (x,y) in L if x >= 0 and x < N and y >= 0 and y < N]

def slope(h,i1,j1,i2,j2):
    return (h[i2,j2] - h[i1,j1]) / (math.sqrt((i2 - i1) ** 2 + (j2 - j1) ** 2))

def dist(i1,j1,i2,j2):
    return math.sqrt((i2 - i1) ** 2 + (j2 - j1) ** 2)

for t in range(num_frames):

    if t % 10 == 0:
        print("Frame %d" % t)
        print_image("out/%d.png" % t, h)
        print("a min %f average %f max %f" % (np.min(a), np.mean(a), np.max(a)))
        print("average lost %f" % (np.mean(np.sqrt(a) * s)))
        print("h min %f max %f" % (np.min(h), np.max(h)))

    # Compute Z and s
    for i in range(N):
        for j in range(N):
            lowerneighs = [(x,y) for (x,y) in neighs(i,j) if h[x,y] < h[i,j]]
            Z[i,j] = 0.
            for (x,y) in lowerneighs:
                Z[i,j] += slope(h, x, y, i, j) ** p

    # Update area 
    iindex = (-h).argsort(axis=None, kind='mergesort')
    jindex = np.unravel_index(iindex, h.shape) 
    index = np.vstack(jindex).T
    for i,j in index:
        upperneighs = [(x,y) for (x,y) in neighs(i,j) if h[x,y] > h[i,j]]
        a[i,j] = 1.
        for (x,y) in upperneighs:
            a[i,j] += (slope(h, i, j, x, y) ** p) / Z[x,y] * a[x,y]

    # Update height
    # h_new = np.copy(h)
    for i,j in reversed(index):
        lowerneighs = [(x,y) for (x,y) in neighs(i,j) if h[x,y] < h[i,j]]
        Z[i,j] = 0.
        s = 0.
        xlow, ylow = i, j
        for (x,y) in lowerneighs:
            snew = slope(h, x, y, i, j)
            if snew > s:
                s = snew
                xlow, ylow = x, y
        if not xlow == i or not ylow == j:
            # implicit
            factor = dt * math.sqrt(a[i,j]) / dist(i, j, xlow, ylow)
            h[i,j] = (h[i,j] + dt * u[i,j] + factor * h[xlow,ylow]) / (1 + factor)

            # explicit
            #h_new[i,j] = max(h[i,j] + dt * (u[i,j] - math.sqrt(a[i,j]) * slope(h, xlow, ylow, i, j)), 0.0)
    #h = h_new

print_image("out/final.png", h)

xs = np.linspace(0, N, N)
ys = np.linspace(0, N, N)
xs, ys = np.meshgrid(xs, ys)
grid = pv.StructuredGrid(xs, ys, (h / h_scale * N * 0.2))
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars=grid.points[:, -1], cmap='gist_earth')
plotter.show_grid()
plotter.show()

