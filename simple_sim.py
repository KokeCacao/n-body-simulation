# -*- coding: utf-8 -*-
"""
N-body problem simulation for matplotlab. Based on
https://fiftyexamples.readthedocs.org/en/latest/gravity.html
Credit: https://github.com/brandones/n-body
"""
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


# Constants and Configurations
G = 6.67428e-11
AU = (149.6e6 * 1000)
ONE_DAY = 24*3600
FILE_NAME = ''
TRACK_NUM = 1
INTERVAL = 1

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
        print("Dumping Parameters of the Latest Run")
        print(json.dumps(data, cls=BodyEncoder, indent=4))

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


def move(bodies, timestep):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pairs = itertools.combinations(bodies, 2)
    
    # Initialize force vectors
    for b in bodies:
        b.f = np.array([0.0, 0.0, 0.0])
    
    # Calculate force vectors
    for b1, b2 in pairs:
        f = b1.attraction(b2)
        b1.f += f
        b2.f -= f
    
    # Update velocities based on force, update positions based on velocity
    # Approximate as linear acceleration
    for body in bodies:
        # v = at = (F/m)t
        body.v += body.f / body.mass * timestep
        # x = vt
        body.p += body.v * timestep

def points_for_bodies(bodies):
    x0 = np.array([body.p[0] for body in bodies])
    y0 = np.array([body.p[1] for body in bodies])
    z0 = np.array([body.p[2] for body in bodies])
    return x0, y0, z0

def norm_forces_for_bodies(bodies, norm_factor):
    u0 = np.array([body.f[0] for body in bodies])
    v0 = np.array([body.f[1] for body in bodies])
    w0 = np.array([body.f[2] for body in bodies])
    return u0/norm_factor, v0/norm_factor, w0/norm_factor


class AnimatedScatter(object):
    def __init__(self, bodies, axis_range, timescale):
        self.bodies = bodies
        self.axis_range = axis_range
        self.timescale = timescale
        self.stream = self.data_stream()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.fig = fig
        self.ax = ax
        self.force_norm_factor = None
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=INTERVAL,
                                           init_func=self.setup_plot, blit=False)

        self.x_ = []
        self.y_ = []
        self.z_ = []

    def setup_plot(self):
        xi, yi, zi, ui, vi, wi, x_, y_, z_ = next(self.stream)

        c = [np.random.rand(3,) for i in range(len(xi))]
        
        self.ax.set_proj_type('ortho')
        self.scatter = self.ax.scatter(xi, yi, zi, c=c, s=10)
        self.quiver = self.ax.quiver(xi, yi, zi, ui, vi, wi, length=1)
        self.lines, = self.ax.plot([], [], [], ".", markersize=0.5)

        self.axtime = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.stime = Slider(self.axtime, 'Time', 0.0, 10.0, valinit=0.0)

        self.resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.button = Button(self.resetax, 'Reset', hovercolor='0.975')

        #Routines to reset and update sliding bar
        def reset(event):
            self.stime.reset()
            self.x_ = []
            self.y_ = []
            self.z_ = []

        def update(val):
            if val == 0: return
            print("Jumping e^{}={} frames".format(int(val), int(math.e**val)))
            for v in range(int(math.e**val)):
                x_i, y_i, z_i, u_i, v_i, w_i, x_, y_, z_  = next(self.stream)
            self.stime.reset()

        #Bind sliding bar and reset button  
        self.stime.on_changed(update)
        self.button.on_clicked(reset)

        FLOOR = self.axis_range[0]
        CEILING = self.axis_range[1]
        self.ax.set_xlim3d(FLOOR, CEILING)
        self.ax.set_ylim3d(FLOOR, CEILING)
        self.ax.set_zlim3d(FLOOR, CEILING)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        return self.scatter, self.quiver

    def quiver_force_norm_factor(self):
        axis_length = np.abs(self.axis_range[1]) + np.abs(self.axis_range[0])
        return np.amax(np.array([b.f for b in self.bodies]))/(axis_length/10)

    def data_stream(self):
        while True:
            move(self.bodies, self.timescale)
            if not self.force_norm_factor:
                self.force_norm_factor = self.quiver_force_norm_factor()
                # print('factor ', self.force_norm_factor)
            x, y, z = points_for_bodies(self.bodies)
            u, v, w = norm_forces_for_bodies(self.bodies, self.force_norm_factor)

            # rad = random.randint(0, TRACK_NUM-1)
            # x_.append(x[rad])
            # y_.append(y[rad])
            # z_.append(z[rad])

            self.x_.append(x[-1])
            self.y_.append(y[-1])
            self.z_.append(z[-1])
            yield x, y, z, u, v, w, self.x_, self.y_, self.z_

    def update(self, i):
        x_i, y_i, z_i, u_i, v_i, w_i, x_, y_, z_  = next(self.stream)

        self.scatter._offsets3d = (x_i, y_i, z_i)

        segments = np.array([(b.p, b.p + b.f/self.force_norm_factor) for b in self.bodies])
        self.quiver.set_segments(segments)

        self.lines.set_data(np.array(x_), np.array(y_))
        self.lines.set_3d_properties(np.array(z_))

        plt.draw()
        return self.scatter, self.quiver, self.lines

    def show(self):
        plt.show()


def parameters_for_simulation(simulation_name):

    if simulation_name == 'solar':
        sun = Body('sun', mass=1.98892 * 10**30, p=np.array([0.0, 0.0, 0.0]))
        earth = Body('earth', mass=5.9742 * 10**24, p=np.array([-1*AU, 0.0, 0.0]), v=np.array([0.0, 29.783 * 1000, 0.0]))
        venus = Body('venus', mass=4.8685 * 10**24, p=np.array([0.723 * AU, 0.0, 0.0]), v=np.array([0.0, -35.0 * 1000, 0.0]))
        axis_range = (-1.2*AU, 1.2*AU)
        timescale = ONE_DAY
        return (sun, earth, venus), axis_range, timescale

    elif simulation_name == 'dump':
        data = json.loads(open(FILE_NAME, "r").read())
        return BodyEncoder.deserialize(data)

    elif simulation_name == 'random_three':
        n = 2
        position = AU/150
        time = 100.
        speed = 0.2
        bodies = []
        for i in range(n):
            name = 'random_{}'.format(n)
            mass = random.uniform(4.0**24,6.0**24)
            p = np.array([random.uniform(-position, position), random.uniform(-position, position), random.uniform(-position, position)])
            v = np.array([random.uniform(-speed, speed),random.uniform(-speed, speed),random.uniform(-speed, speed)])
            # v = np.array([0.,0.,0.])
            bodies.append(Body(name, mass=mass, p=p, v=v))
            # print("Appending {}, {}, {}, {}".format(name, mass, p, v))
        bodies.append(Body("sun", mass=5.0**24, p=np.array([0.,0.,0.])))

        axis_range = (-2*position, 2*position)
        timescale = ONE_DAY*time
        return tuple(bodies), axis_range, timescale

    elif simulation_name == 'random':
        n = 49
        position = AU/150
        time = 100.
        speed = 2.
        bodies = []
        for i in range(n):
            name = 'random_{}'.format(n)
            mass = random.uniform(4.0**24,6.0**24)
            p = np.array([random.uniform(-position, position), random.uniform(-position, position), random.uniform(-position, position)])
            v = np.array([random.uniform(-speed, speed),random.uniform(-speed, speed),random.uniform(-speed, speed)])
            # v = np.array([0.,0.,0.])
            bodies.append(Body(name, mass=mass, p=p, v=v))
            # print("Appending {}, {}, {}, {}".format(name, mass, p, v))
        bodies.append(Body("sun", mass=6.5**24, p=np.array([0.,0.,0.])))

        axis_range = (-2*position, 2*position)
        timescale = ONE_DAY*time
        return tuple(bodies), axis_range, timescale
    
    elif simulation_name == 'negative_random':
        n = 20
        position = AU
        time = 10.
        speed = 1000.
        bodies = []
        for i in range(n):
            name = 'random_{}'.format(n)
            mass = random.uniform(-1.0**23,-2.0**24)
            p = np.array([random.uniform(-position, position), random.uniform(-position, position), random.uniform(-position, position)])
            v = np.array([random.uniform(-speed, speed),random.uniform(-speed, speed),random.uniform(-speed, speed)])
            # v = np.array([0.,0.,0.])
            bodies.append(Body(name, mass=mass, p=p, v=v))
            # print("Appending {}, {}, {}, {}".format(name, mass, p, v))
        bodies.append(Body("sun", mass=-9.0**24, p=np.array([0.,0.,0.])))

        axis_range = (-2*AU, 2*AU)
        timescale = ONE_DAY*time
        return tuple(bodies), axis_range, timescale
    
    elif simulation_name == 'mix_random':
        n = 20
        position = AU
        time = 10.
        speed = 1000.
        bodies = []
        for i in range(n):
            name = 'random_{}'.format(n)
            mass = random.choice([random.uniform(-1.0**21,-2.0**20), random.uniform(1.0**23,2.0**24), random.uniform(1.0**23,2.0**24), random.uniform(1.0**23,2.0**24)])
            p = np.array([random.uniform(-position, position), random.uniform(-position, position), random.uniform(-position, position)])
            v = np.array([random.uniform(-speed, speed),random.uniform(-speed, speed),random.uniform(-speed, speed)])
            # v = np.array([0.,0.,0.])
            bodies.append(Body(name, mass=mass, p=p, v=v))
            # print("Appending {}, {}, {}, {}".format(name, mass, p, v))
        bodies.append(Body("sun", mass=-9.0**24, p=np.array([0.,0.,0.])))

        axis_range = (-2*AU, 2*AU)
        timescale = ONE_DAY*time
        return tuple(bodies), axis_range, timescale

    elif simulation_name == 'misc3':
        """
        This is basically mostly just a demonstration that this
        simulation doesn't respect conservation laws.
        """
        sun1 = Body('A', mass=6.0 * 10**30, p=np.array([4.0*AU, 0.5*AU, 0.0]), v=np.array([-10.0 * 1000, -1.0 * 1000, 0.0]))
        sun2 = Body('B', mass=8.0 * 10**30, p=np.array([-6.0*AU, 0.0, 3.0*AU]), v=np.array([5.0 * 1000, 0.0, 0.0]))
        sun3 = Body('C', mass=10.0 * 10**30, p=np.array([0.723 * AU, -5.0 * AU, -1.0*AU]), v=np.array([-10.0 * 1000, 0.0, 0.0]))
        axis_range = (-20*AU, 20*AU)
        timescale = ONE_DAY
        return (sun1, sun2, sun3), axis_range, timescale

    elif simulation_name == 'centauri3':
        """
        Based on known orbit dimensions and masses.
        Not working, not sure why. They shouldn't get farther than 36AU
        or about 5e12m away from each other.
        """
        p_a = np.array([-7.5711 * 10**11, 0.0, 0.0])
        p_b = np.array([9.1838 * 10**11, 0.0, 0.0])
        v_a = np.array([0.0, 1.212 * 10**4, 0.0])
        v_b = np.array([0.0, -1.100 * 10**4, 0.0])
        alphaA = Body('Alpha A', mass=1.100*1.98892 * 10**30, p=p_a, v=v_a)
        alphaB = Body('Alpha B', mass=0.907*1.98892 * 10**30, p=p_b, v=v_b)
        axis_range = (-10.0**13, 10.0**13)
        timescale = ONE_DAY * 5
        return (alphaA, alphaB), axis_range, timescale

    else:
        raise ValueError('No simulation named {}'.format(simulation_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', help='which configureation to use')
    parser.add_argument('--load', dest='load', help='which configureation to use')
    args = parser.parse_args()
    global FILE_NAME
    print("args.mode = {}".format(args.mode))
    print("args.load = {}".format(args.load))
    FILE_NAME= args.load if args.load is not None else FILE_NAME

    bodies, axis_range, timescale = parameters_for_simulation(args.mode)

    data = {"bodies": bodies, "axis_range": axis_range, "timescale": timescale}
    BodyEncoder.serialize(data)

    a = AnimatedScatter(bodies, axis_range, timescale)
    a.show()

if __name__ == '__main__':
    main()
