#!/usr/bin/python

import scipy.spatial
import numpy as np
import random



def dist(p1, p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))


class RRT:
    def __init__(self, start_point, bounds_min, bounds_max):
        self.nodes = [start_point]
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.d = len(start_point)
        
        assert(len(bounds_max) == len(bounds_min))
        assert(len(start_point) == len(bounds_min))

    def sample(self):
        sampled_p = []
        for ind in range(self.d):
            lower = self.bounds_min[ind]
            upper = self.bounds_max[ind]
            sampled_p.append(random.random()*(upper-lower) + lower)
        return np.array(sampled_p)

    def find_nearest(self, s):
        cur_closest = self.nodes[0]
        cur_dist = dist(s, cur_closest)

        for node in self.nodes:
            if dist(s, node) < cur_dist:
                cur_dist = dist(s, node)
                cur_closest = node
        return cur_closest

    def step(self):
        """Extends the tree, returning the (node, new_sample) path that connects the new sample"""
        
        s = self.sample()
        nearest = self.find_nearest(s)
        
        self.nodes.append(s)
        return (nearest, s)
