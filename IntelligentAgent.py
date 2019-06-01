from agent import Agent
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation

class IntelligentAgent():

    def __init__(self, conn_index, util_matrix):
        #agent index
        self.conn_index = conn_index
        self.util_matrix = util_matrix

    def MEU(self, conn_matrix, agents_decisions):
        """
        """
        utilities = []
        m, n = self.util_matrix.shape
        agent_count = len(conn_matrix[0, :])

        for decision_idx in range(m):
            utility = 0
            for i, conn_strength in enumerate(conn_matrix[self.conn_index, :]):
                other_agent_decision_idx = agents_decisions[i]
                utility += conn_strength * self.util_matrix[decision_idx, other_agent_decision_idx]

            utilities.append(utility)
        return np.argmax(utilities)

def update_agents_decision(conn_matrix, agents, agents_decisions):
    new_agents_decisions = []
    for i, decision in enumerate(agents_decisions):    
        new_agents_decisions.append(
            agents[i].MEU(conn_matrix, agents_decisions)
        )
    #sanity check for animation update
    #new_agents_decisions[2] = np.random.randint(2, size=10)
    return new_agents_decisions


def animate(conn_matrix, agents, agents_decisions):
    G = nx.from_numpy_matrix(conn_matrix)
    pos = nx.spring_layout(G)
    n = len(conn_matrix[:, 0])
    fig, ax = plt.subplots(figsize=(6,4))
    
    def update(num):
        ax.clear()
        nonlocal agents_decisions
        color_map = []
        for x in agents_decisions:
            if x == 0: 
                color_map.append("red")
            elif x == 1:
                color_map.append("blue")
            else:
                color_map.append("white")
    
        nx.draw(
                G, pos, edge_color='black',width=1,linewidths=1,\
                node_size=500, node_color = color_map , alpha=0.9,\
                labels={node:node for node in G.nodes()}
                )

        edge_labels = {(j, i):conn_matrix[i, j] for j in range(n) for i in range(n)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

        agents_decisions = update_agents_decision(conn_matrix, agents, agents_decisions)

    ani = matplotlib.animation.FuncAnimation(fig, update, interval=1000, repeat=True)
    plt.axis('off')
    plt.show()


def test():
    conn_matrix = np.array([
        [0, 2, 1], #ballot cast for self so inf
        [2, 0, 3],  #influence/conn on self is zero since undecided
        [1, 3, 0] 
    ])
    agents_decisions = np.array([
         0, #dem vote
        -1, #curr agent undecided -- doenst really since self influence is zero on conn_matrix
         1  #rep vote
    ])
    dominator = 100000
    #util matrix doesnt need to take into account utility of other agent
    #and so only needs be a 2x2 matrix
    und_util_matrix = np.array([
        [2, 1],
        [1, 1]
    ])
    rep_util_matrix = np.array([
        [0, 0],
        [dominator, dominator]
    ])
    dem_util_matrix = np.array([
        [dominator, dominator],
        [0, 0]
    ])

    agent_0 = IntelligentAgent(0, dem_util_matrix)
    agent_1 = IntelligentAgent(1, und_util_matrix)
    agent_2 = IntelligentAgent(2, rep_util_matrix)

    agents = [agent_0, agent_1, agent_2]

    assert agent_1.MEU(conn_matrix, agents_decisions) == 0
    assert np.allclose(update_agents_decision(conn_matrix, agents, agents_decisions), np.array([0, 0, 1]))
    print("SUCCESS: MEU")

    animate(conn_matrix, agents, agents_decisions)


test()