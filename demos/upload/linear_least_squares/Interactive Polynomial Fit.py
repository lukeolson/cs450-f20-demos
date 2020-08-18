#!/usr/bin/env python
# coding: utf-8

# # Interactive Polynomial Fitting

# This notebook is best experienced when exported to a Python script and run from the command line.

# In[3]:


import numpy as np
import numpy.linalg as la

from matplotlib.pyplot import (
    clf, plot, show, xlim, ylim,
    get_current_fig_manager, gca, draw, connect)


# Run this cell to play with the node placement toy:

# In[4]:


x_points = []
y_points = []
deg = [1]

def update_plot():
    clf()
    xlim([-1, 1])
    ylim([-1.5, 1.5])
    gca().set_autoscale_on(False)
    plot(x_points, y_points, 'o')

    if len(x_points) >= deg[0]+1:
        eval_points = np.linspace(-1, 1, 500)
        poly = np.poly1d(np.polyfit(
            np.array(x_points),
            np.array(y_points), deg[0]))
        plot(eval_points, poly(eval_points), "-")


def click(event):
    tb = get_current_fig_manager().toolbar
    if event.button == 1 and event.inaxes and tb.mode == '':
        x_points.append(event.xdata)
        y_points.append(event.ydata)

    if event.button == 3 and event.inaxes and tb.mode == '':
        if len(x_points) >= deg[0]+2:
            deg[0] += 1

    update_plot()
    draw()

update_plot()
connect('button_press_event', click)
show()


# In[ ]:




