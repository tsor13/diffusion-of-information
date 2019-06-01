from graph import Graph
from agent import Agent
import numpy as np
import pdb
import networkx as nx
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation

Q_DEFAULT = np.array([[.5,.25,.25],[.25,.5,.25],[.25,.25,.5]])
# 1-cycle
def one_cycle_graph(q_matrix = Q_DEFAULT, diffusion=True):
    '''Creates and returns Graph object using
       the given 1-cycle adjacency matrix.
    '''

    #adjacency matrix
    matrix1 = np.array([[0,1,0,0,0,0,1],
                        [1,0,1,0,0,0,0],
                        [0,1,0,1,0,0,0],
                        [0,0,1,0,1,0,0],
                        [0,0,0,1,0,1,0],
                        [0,0,0,0,1,0,1],
                        [1,0,0,0,0,1,0]])

    #initialize node vector of policy 0 (i.e. hare)
    if diffusion:
        nodes1 = np.full(matrix1.shape[0], np.zeros_like(matrix1.shape[0]))

    #initialize node vector of Agent objects
    else:
        nodes1 = np.full(matrix1.shape[0], create_agent_list(matrix1.shape[0],q_matrix))

    #return Graph object, which takes an adjacency matrix
    # and Agent node-vector as parameters
    return Graph(matrix1, nodes1)

# 2-cycle
def two_cycle_graph(q_matrix = Q_DEFAULT, diffusion=True):
    '''Creates and returns Graph object using
       the given 2-cycle adjacency matrix.
    '''

    #adjacency matrix
    matrix2 = np.array([[0,1,1,0,0,1,1],
                        [1,0,1,1,0,0,1],
                        [1,1,0,1,1,0,0],
                        [0,1,1,0,1,1,0],
                        [0,0,1,1,0,1,1],
                        [1,0,0,1,1,0,1],
                        [1,1,0,0,1,1,0]])

    #initialize node vector of policy 0 (i.e. hare)
    if diffusion:
        nodes2 = np.full(matrix2.shape[0], np.zeros_like(matrix2.shape[0]))

    #initialize node vector of Agent objects
    else:
        nodes2 = np.full(matrix2.shape[0], create_agent_list(matrix2.shape[0],q_matrix))

    #return Graph object, which takes an adjacency matrix
    # and Agent node-vector as parameters
    return Graph(matrix2, nodes2)

# complete
def complete_graph(q_matrix = Q_DEFAULT, diffusion=True):
    '''Creates and returns Graph object using
       the given 2-cycle adjacency matrix.
    '''

    #adjacency matrix
    matrixc = np.array([[0,1,1,1,1,1,1],
                        [1,0,1,1,1,1,1],
                        [1,1,0,1,1,1,1],
                        [1,1,1,0,1,1,1],
                        [1,1,1,1,0,1,1],
                        [1,1,1,1,1,0,1],
                        [1,1,1,1,1,1,0]])

    #initialize node vector of policy 0 (i.e. hare)
    if diffusion:
        nodesc = np.full(matrixc.shape[0], np.zeros_like(matrixc.shape[0]))

    #initialize node vector of Agent objects
    else:
        nodesc = np.full(matrixc.shape[0], create_agent_list(matrixc.shape[0],q_matrix))

    #return Graph object, which takes an adjacency matrix
    # and Agent node-vector as parameters
    return Graph(matrixc, nodesc)


# ad hoc
def ad_hoc_graph(q_matrix = Q_DEFAULT, diffusion=True):
    '''Creates and returns Graph object using
       the given adhoc adjacency matrix.
    '''

    #adjacency matrix
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

    #initialize node vector of policy 0 (i.e. hare)
    if diffusion:
        nodes_adhoc = np.full(adhoc.shape[0], np.zeros_like(adhoc.shape[0]))

    #initialize node vector of Agent objects
    else:
        nodes_adhoc = np.full(adhoc.shape[0], create_agent_list(adhoc.shape[0],q_matrix))

    #return Graph object, which takes an adjacency matrix
    # and Agent node-vector as parameters
    return Graph(adhoc, nodes_adhoc)


def create_agent_list(n,q_matrix = None):
    '''Create a list of Agent objects of length n
       for use in initializing a Graph object.
    '''
    # Default q matrix is .4 on diagonals, .3 elsewhere
    if q_matrix is None:
        q_matrix = np.array([[.4,.3,.3],[.3,.4,.3],[.3,.3,.4]])

    return [Agent(q_matrix) for _ in range(n)]

def test_information_cascade(g,filename=None,first_agent_num=None,well_position=None):
    """ Run a day's worth of agent decisions """
    # If no well position or first agent is given, pick one randomly
    if well_position == None:
        well_position = np.random.choice([0,1,2])

    if first_agent_num == None:
        first_agent_num = np.random.choice(list(range(g.n)))

    agents_left = set(range(g.n))
    action_queue = [first_agent_num]
    agents_left.remove(first_agent_num)
    action_order = []
    while(len(action_queue) > 0):
        agent_num = action_queue.pop(0)
        g.node_act_information_cascade(agent_num, well_position)
        action_order.append(agent_num)
        # Add the un-tagged neighbors to the action queue
        for neighbor in g.adjacent_nodes(agent_num):
            if neighbor in agents_left:
                action_queue.append(neighbor)
                agents_left.remove(neighbor)


    # set colors for each available action.
    # Tracks a full color map for each node after each agent decision
    c_map = ['gray'] * g.n
    color_maps = [c_map ]

    for action, order in zip(g.actions, action_order):

        c_map[int(order)] = ['cyan','orange','lime'][int(action)] # agent visits well 0, 1, or 2
        color_maps.append(c_map.copy())

    fig, ax = plt.subplots(figsize=(6,4))
    G = nx.Graph(g.matrix)
    pos = nx.spring_layout(G)
    def update(num):
        ax.clear()
        plt.title(f"Starting Point: {first_agent_num}\nCorrect Well:{well_position}")
        well_0 = mpatches.Patch(color='cyan', label='Well 0')
        well_1 = mpatches.Patch(color='orange', label='Well 1')
        well_2 = mpatches.Patch(color='lime', label='Well 2')
        plt.legend(handles=[well_0, well_1, well_2])

        nx.draw(G,pos=pos,node_color=color_maps[(num+1) % len(color_maps)],with_labels = True)

    # create and draw networkx graph

    ani = matplotlib.animation.FuncAnimation(fig, update, interval=1000, repeat=True)
    # Uncomment to allow saving animations in .gif format
    # if filename:
    #     ani.save(filename)
    plt.show()

    #return the list of actions
    return g.actions


def test_diffusion(g,first_agent_num=None):
    """ Run a day's worth of agent decisions """

    if first_agent_num == None:
        first_agent_num = np.random.choice(list(range(g.n)))

    agents_left = set(range(g.n))
    action_queue = g.adjacent_nodes(first_agent_num)
    [agents_left.remove(n) for n in g.adjacent_nodes(first_agent_num)]

    while(len(action_queue) > 0):
        agent_num = action_queue.pop(0)
        g.node_act_diffusion(agent_num)

        # Add the un-tagged neighbors to the action queue
        for neighbor in g.adjacent_nodes(agent_num):
            if neighbor in agents_left:
                action_queue.append(neighbor)
                agents_left.remove(neighbor)


    #set colors for each available action
    color_map = []
    for action in g.actions:
        color_map.append(['cyan','orange'][action]) # agent visits well 0, 1, or 2

    #create and draw networkx graph
    plt.title('Starting Point:' + str(first_agent_num))
    G = nx.Graph(g.matrix)
    nx.draw(G,node_color=color_map,with_labels=True)
    stag = mpatches.Patch(color='cyan', label='Stag')
    hare = mpatches.Patch(color='orange', label='Hare')

    plt.legend(handles=[stag, hare])
    plt.show()

    #return the list of actions
    return g.actions



###  TESTING  ###
def test_IC():
    # This SHOULD illustrate an information cascade in each Graph g
    print('Information Cascade')

    # We will iterate through this graph to test diffusion
    graph_list = [one_cycle_graph(diffusion=False),
                  two_cycle_graph(diffusion=False),
                  complete_graph(diffusion=False),
                  ad_hoc_graph(diffusion=False)]

    updating_list = []
    names = ['output/one_cycle_cascade.gif','output/two_cycle_cascade.gif','output/complete_cycle_cascade.gif','output/ad_hoc_cascade.gif']
    for name,g in zip(names, graph_list):
        updating_list.append(test_information_cascade(g,filename = name))


    ##uncomment to test on a single graph
    #a2 = test_information_cascade(two_cycle_graph())

def test_DI():
    print('Diffusion of Innovation')

    graph_list = [one_cycle_graph(),
                  two_cycle_graph(),
                  complete_graph(),
                  ad_hoc_graph()]

    updating_list = []
    for g in graph_list:
        updating_list.append(test_diffusion(g,first_agent_num=0))
