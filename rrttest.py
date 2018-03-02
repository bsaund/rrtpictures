#!/usr/bin/python

from rrt import *
import IPython

r = RRT([0,0], [-10,-10], [10,10])

r.step()

IPython.embed()
