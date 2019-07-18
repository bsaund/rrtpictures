#!/usr/bin/python

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from matplotlib import pyplot as plt
import IPython


def extract_solution(manager, routing, assignment):
    index = routing.Start(0)
    path = []
    while not routing.IsEnd(index):
        path.append(manager.IndexToNode(index))
        index = assignment.Value(routing.NextVar(index))
    return path


def tsp(points):
    """
    returns the travelling salesman shortest path
    points: list of points np.array([[2,1], [3,5], ...])
    """
    manager = pywrapcp.RoutingIndexManager(
        len(points), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(np.linalg.norm(points[from_node] - points[to_node]))

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    print("Solving")
    assignment = routing.SolveWithParameters(search_parameters)
    print("Solved")

    if not assignment:
        print("Not assignment")
        
    return extract_solution(manager, routing, assignment)


def test_points():
    points = [[100*np.sin(i), 100*np.cos(i)] for i in np.arange(0, np.pi*2, 0.1)]
    np.random.shuffle(points)
    return np.array(points)

if __name__ == "__main__":
    points = test_points()
    # IPython.embed()
    path = tsp(points)
    print(path)
    # IPython.embed()
    x = []
    y = []
    # IPython.embed()
    for p in path:
        print(points[p])
        x.append(points[p,0])
        y.append(points[p,1])


    plt.plot(x, y)
    plt.show()
