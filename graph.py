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

    def node_act_information_cascade(self, agent_num, current_well):
        """ has node agent_num act and stores the action """
        #   Get adjacent actions
        neighbor_observations = self.parse_adjacent_actions( self.adjacent_actions(agent_num) )
        action = self.nodes[agent_num].act_information_cascade(current_well, neighbor_observations)
        self.actions[agent_num] = action
        return action

    def adjacent_nodes(self, agent_num):
        """ returns list of adjacent nodes """
        adj = []
        row = self.matrix[agent_num]
        for i in range(self.n):
            if row[i] == 1:
                adj.append(i)
        return adj

    def adjacent_actions(self, agent_num):
        """ returns list of actions taken by adjacent nodes """
        adjacent_actions = []
        adj_nodes = self.adjacent_nodes(agent_num)

        for node in adj_nodes:
            action = self.actions[node]
            if action != -1:
                adjacent_actions.append(action)
        return adjacent_actions

    def parse_adjacent_actions(self, actions):
        """ Reads in a list of actions {0,1,2} and returns a 3-tuple with the
        number of each instance. Does not factor in default -1 values for agents
        who have not taken an action """
        actions = np.array(actions)
        return [np.count_nonzero(actions == i) for i in range(3)]
