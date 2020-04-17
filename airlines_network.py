#! /usr/bin/python3

import random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ============== read in data from a file ==============

df = open("airlines.txt", 'r')
num_vertices = 0

for line in df:
    line_list = line.strip().split()
    if line_list[0] == '*Vertices':
        num_vertices = int(line_list[1])
        break

print("num_vertices =", num_vertices)
G = nx.empty_graph(num_vertices)

reading_edges = False
for line in df:
    line_list = line.strip().split()
    if not reading_edges:
        if line_list[0] != "*Edges":
            continue
        else:
            reading_edges = True
            continue
    else:
        G.add_edge(int(line_list[0]), int(line_list[1]))


ne = G.size()
print("num_edges =", ne)