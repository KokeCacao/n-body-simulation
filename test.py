import argparse
import random
import itertools
import json
from math import atan2, sin, cos
from time import sleep

import numpy as np
import math

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


G = 6.67428e-11
AU = (149.6e6 * 1000)
ONE_DAY = 24*3600
TRACK_NUM = 1

# Serialization Utility
class BodyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, '__jsonencode__'):
            return obj.__jsonencode__()
        if isinstance(obj, set):
            return list(obj)
        return obj.__dict__
    
    def deserialize(data):
        bodies = [Body(d["name"],d["mass"],np.array(d["p"]),np.array(d["v"])) for d in data["bodies"]]
        axis_range = data["axis_range"]
        timescale = data["timescale"]
        return bodies, axis_range, timescale

    def serialize(data):
        file = open('data.json', 'w+')
        json.dump(data, file, cls=BodyEncoder, indent=4)
        return json.dumps(data, cls=BodyEncoder, indent=4)

# Body Object
class Body(object):
    def __init__(self, name, mass, p, v=(0.0, 0.0, 0.0)):
        self.name = name
        self.mass = mass
        self.p = p
        self.v = v
        self.f = np.array([0.0, 0.0, 0.0])

    def __str__(self):
        return 'Body {}'.format(self.name)

    def attraction(self, other):
        """Calculate the force vector between the two bodies"""
        assert self is not other
        diff_vector = other.p - self.p
        distance = norm(diff_vector)

        # Remove Collision
        # assert np.abs(distance) < 10**4, 'Bodies collided!'

        # F = GMm/r^2
        f_tot = G * self.mass * other.mass / (distance**2)
        # Get force with a direction
        f = f_tot * diff_vector / distance
        return f
    
    # def __dict__(self):
    #     return {"name": self.name, "mass": self.mass, "p": self.p.tolist(), "v": self.v.tolist(), "f": self.f.tolist()}


def norm(x):
    """return: vector length in n demension"""
    return np.sqrt(np.sum(x**2))
# class LineBuilder:
#     def __init__(self, fig, line, scatter):
#         self.fig = fig
#         self.line = line
#         self.scatter = scatter
        
#         self.xs = list(line.get_xdata())
#         self.ys = list(line.get_ydata())
        
#         self.xd = []
#         self.yd = []
        
        
#         self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        
#         self.create = True

#     def __call__(self, event):
#         print('click', event)
#         if event.inaxes!=self.line.axes: return
        
#         if self.create:
#             self.xd.append(event.xdata)
#             self.yd.append(event.ydata)

#             self.scatter.set_offsets(np.c_[self.xd,self.yd])
#     #         self.fig.canvas.draw_idle()

#         else:
#             self.line.set_data(self.xs, self.ys)
#             self.line.figure.canvas.draw()
        
#         self.create = not self.create
        

# # init plots
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('click to build line segments')

# # init objects
# line, = ax.plot([0], [0])  # empty line
# scatter = ax.scatter([0], [0], s=10)

# linebuilder = LineBuilder(fig, line, scatter)

# plt.show()
class LineBuilder:
    def __init__(self, fig, vector, scatter):
        self.fig = fig
        self.vector = vector
        self.scatter = scatter
        
        self.xv = []
        self.yv = []
        
        self.xd = []
        self.yd = []
        
        
        self.cid = vector.figure.canvas.mpl_connect('button_press_event', self)
        
        self.create = True
        
        
        self.n = 3 ## HOW MANY PLAYERS HERE
        
        self.random = []
        for i in range(self.n):
            s = random.uniform(4.0**24,6.0**24)
            self.random.append(s)
            print(s)
        
        
        self.current_n = 0
        self.position = AU/150
        self.time = 100.
        self.speed = 0.2
        self.bodies = []
        self.axis_range = (-2*self.position, 2*self.position)
        self.timescale = ONE_DAY*self.time


    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.vector.axes: return
        
        if self.create:
            self.xd.append(event.xdata)
            self.yd.append(event.ydata)

            self.scatter.set_offsets(np.c_[self.xd,self.yd])
    #         self.fig.canvas.draw_idle()
            data = [event.xdata]

        else:
            self.xv.append(event.xdata)
            self.yv.append(event.ydata)
#             self.vector.set_data(self.xv, self.yv)
#             self.vector.figure.canvas.draw()
            self.vector.set_offsets(np.c_[self.xv,self.yv])
    
            if self.current_n <= self.n:
                mass = self.random[self.current_n]
                name = '{}'.format(self.current_n)
                p = [self.xd[-1], self.yd[-1], 0]
                v = ([(self.xv[-1] - p[0])/10e8, (self.yv[-1] - p[1])/10e8, 0])
                self.bodies.append(Body(name, mass=mass, p=p, v=v))
                if self.current_n == self.n:
                    data = {"bodies": tuple(self.bodies), "axis_range": self.axis_range, "timescale": self.timescale}
                    print(BodyEncoder.serialize(data))
                
                self.current_n = self.current_n + 1
            else:
                print("STOP")
                    
        
        self.create = not self.create
        

# init plots
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('click to build line segments')

ax.set_xlim([-2e9,2e9])
ax.set_ylim([-2e9,2e9])

# init objects
vector = ax.scatter([0], [0], s=2)
scatter = ax.scatter([0], [0], s=20)

linebuilder = LineBuilder(fig, vector, scatter)

plt.grid()
plt.show()