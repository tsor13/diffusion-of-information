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

    def run_day(self, first_agent_num = None, well_position = None):
        """ Run a day's worth of agent decisions """
        # If no well position or first agent is given, pick one randomly
        if well_position == None:
            well_position = np.random.choice([0,1,2])

        if first_agent_num == None:
            first_agent_num = np.random.choice(list(range(7)))

        agents_left = set(range(self.n))
        action_queue = [first_agent_num]
        agents_left.remove(first_agent_num)

        while(len(action_queue) > 0):
            agent_num = action_queue.pop(0)
            self.node_act_information_cascade(agent_num, well_position)

            # Add the un-tagged neighbors to the action queue
            for neighbor in self.adjacent_nodes(agent_num):
                if neighbor in agents_left:
                    action_queue.append(neighbor)
                    agents_left.remove(neighbor)


        print(self.actions)
        return

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
        return [np.count_nonzero(actions == i) for i in range(3)]
