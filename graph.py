####################################
import numpy as np

class Graph:
    # CLASS FOR REPRESENTING GRAPH OBJECT

    def __init__(self, adjacency_matrix, nodes):
        # input:
        # adjacency_matrix: n x n numpy matrix of graph topology
        #                   0 for not adjacent, 1 for adjacent
        # nodes: list of nodes in graph of type Agent
        #        for node_act to function right, each agent must have
        #        an act function defined
        self.matrix = adjacency_matrix
        self.n = adjacency_matrix.shape[0]
        assert self.n == adjacency_matrix.shape[1]

        # holds all graph nodes, same indices as adjacency matrix
        self.nodes = np.array(nodes)
        assert self.n == self.nodes.size

        # holds all previous decisions of agents. -1 is default (no action)
        self.actions = np.full(self.n, -1)

    def node_act(self, agent_num):
        # has node agent_num act and stores the action
        action = self.nodes[agent_num].act()
        self.actions[agent_num] = action
        return action

    def adjacent_nodes(self, agent_num):
        # returns list of adjacent nodes
        adj = []
        row = self.matrix[agent_num]
        for i in range(self.n):
            if row[i] == 1:
                adj.append(i)
        return adj

    def adjacent_actions(self, agent_num):
        # returns list of actions taken by adjacent nodes
        adjacent_actions = []
        adj_nodes = self.adjacent_nodes(agent_num)
        for node in adj_nodes:
            action = self.actions[node]
            if action != -1:
                adjacent_actions.append(action)
        return adjacent_actions
