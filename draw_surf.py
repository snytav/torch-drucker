from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import matplotlib.pyplot as plt

def surf(X,Y,f,tit):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Plot the surface.
    surf = ax.plot_surface(X, Y, f.T, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.title(tit)
    plt.xlabel('X')
    plt.ylabel('V')
    fig.colorbar(surf)
    plt.savefig(tit+'.png')

def contour(X,Y,f,tit):
    plt.figure()
    plt.contourf(X, Y, f.T, 50, cmap='coolwarm')
    plt.title(tit)
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('V')
    plt.savefig(tit+'.png')