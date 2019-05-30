from graph import Graph
from agent import Agent
from agent_factory import AgentFactory
import numpy as np
import pdb


factory = AgentFactory()
# 1-cycle
matrix1 = np.array([[0,1,0,0,0,0,1],
                    [1,0,1,0,0,0,0],
                    [0,1,0,1,0,0,0],
                    [0,0,1,0,1,0,0],
                    [0,0,0,1,0,1,0],
                    [0,0,0,0,1,0,1],
                    [1,0,0,0,0,1,0]])
well_solution = 0
# add agents as nodes
n = matrix1.shape[0]
nodes1 = []
# example nodes matrix
for i in range(n):
    good_agent = factory.create_informed_agent(well_solution, False)
    nodes1.append(good_agent)
# change agent 1 to bad agent
bad_agent = factory.create_uninformed_agent()
nodes1[1] = bad_agent

# create graph
g1 = Graph(matrix1, nodes1)

##############################
# Example code for having agents act and update info
# have node 0 act
action0 = g1.node_act(0)
print(action0)
# update node 1 dist_params
actions = g1.adjacent_actions(1)
g1.nodes[1].update_dist_params(actions,c=.1)
g1.node_act(1)
# example data
print(g1.matrix)
print(g1.adjacent_nodes(0))
print(g1.actions)
##############################

# 2-cycle
matrix2 = np.array([[0,1,1,0,0,1,1],
                    [1,0,1,1,0,0,1],
                    [1,1,0,1,1,0,0],
                    [0,1,1,0,1,1,0],
                    [0,0,1,1,0,1,1],
                    [1,0,0,1,1,0,1],
                    [1,1,0,0,1,1,0]])
nodes2 = np.full(matrix2.shape[0], bad_agent)
g2 = Graph(matrix2, nodes2)

# complete
matrixc = np.array([[0,1,1,1,1,1,1],
                    [1,0,1,1,1,1,1],
                    [1,1,0,1,1,1,1],
                    [1,1,1,0,1,1,1],
                    [1,1,1,1,0,1,1],
                    [1,1,1,1,1,0,1],
                    [1,1,1,1,1,1,0]])
nodesc = np.full(matrixc.shape[0], bad_agent)
gc = Graph(matrixc, nodesc)

# ad hoc
adhoc = np.zeros((17,17))
adhoc[0,[1,2]] = 1
adhoc[1,[0,2]] = 1
adhoc[2,[0,1,3]] = 1
adhoc[3,[2,4,8]] = 1
adhoc[4,[3,5,6]] = 1
adhoc[5,[4,6,7]] = 1
adhoc[6,[4,5,7,8,9]] = 1
adhoc[7,[5,6,9,16]] = 1
adhoc[8,[3,6,9,10]] = 1
adhoc[9,[6,7,8,11]] = 1
adhoc[10,[8,11,12]] = 1
adhoc[11,[9,10,13,14]] = 1
adhoc[12,[10,13]] = 1
adhoc[13,[11,12,14,15]] = 1
adhoc[14,[11,13,15,16]] = 1
adhoc[15,[13,14,16]] = 1
adhoc[16,[7,14,15]] = 1

nodes_adhoc = np.full(matrixc.shape[0], bad_agent)
g_adhoc = Graph(adhoc, nodes_adhoc)
print(g_adhoc.adjacent_nodes(11))
