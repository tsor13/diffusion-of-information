import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation

matrix1 = np.array([[0,1,0,0,0,0,1],
                    [1,0,1,0,0,0,0],
                    [0,1,0,1,0,0,0],
                    [0,0,1,0,1,0,0],
                    [0,0,0,1,0,1,0],
                    [0,0,0,0,1,0,1],
                    [1,0,0,0,0,1,0]])

matrix2 = np.array([[0,1,1,0,0,1,1],
                    [1,0,1,1,0,0,1],
                    [1,1,0,1,1,0,0],
                    [0,1,1,0,1,1,0],
                    [0,0,1,1,0,1,1],
                    [1,0,0,1,1,0,1],
                    [1,1,0,0,1,1,0]])

matrixc = np.array([[0,1,1,1,1,1,1],
                    [1,0,1,1,1,1,1],
                    [1,1,0,1,1,1,1],
                    [1,1,1,0,1,1,1],
                    [1,1,1,1,0,1,1],
                    [1,1,1,1,1,0,1],
                    [1,1,1,1,1,1,0]])

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


class DiffusionGraph():
    DEFAULT_UTILITY = np.array([[1,0],[1,2]])

    def __init__(self,adjacency_matrix):
        self.matrix = adjacency_matrix
        self.n = len(self.matrix)
        self.nodes = np.zeros(self.n)

    def get_adjacent_values(self, agent_num):
        adj = np.zeros(2)
        for neighbor in np.where(self.matrix[agent_num] == 1)[0]:
            adj[int(self.nodes[neighbor])] += 1

        return adj

    def run(self,utility_matrix = DEFAULT_UTILITY,early_adopters = [0]):
        # Reset
        self.nodes = np.zeros(self.n)
        # Set up with early adopters
        for i in early_adopters:
            self.nodes[i] = 1


        choices = [self.nodes.copy()]
        # Run until equilibrium is reached
        while(True):
            this_round = np.zeros(self.n)
            for agent in range(self.n):
                if agent in early_adopters:
                    print(agent, "doesnt change, they are an early adopter")
                    this_round[agent] = 1
                    continue
                adj = self.get_adjacent_values(agent)
                p = adj @ utility_matrix
                print(f"agent {agent} has adj {adj}, p {p}")
                # Default to switching in case of a tie
                print(agent)
                this_round[agent] = 0 if p[1] < p[0] else 1


            choices.append(this_round)
            print(this_round)
            if np.allclose(this_round, choices[-2]):
                break
            self.nodes = this_round


        # Animate
        fig, ax = plt.subplots(figsize=(6,4))
        G = nx.Graph(self.matrix)
        pos = nx.spring_layout(G)
        print('choices')
        print(choices)
        def update(num):
            frame = num % len(choices)
            print(frame)
            ax.clear()
            plt.title(f"Early Adopters: {early_adopters}")
            hare = mpatches.Patch(color='gray', label='Hare')
            stag = mpatches.Patch(color='cyan', label='Stag')
            plt.legend(handles=[hare, stag])
            colors = [['gray','cyan'][int(i)] for i in choices[frame]]
            nx.draw(G,pos=pos,node_color=colors,with_labels = True)

        # create and draw networkx graph
        ani = matplotlib.animation.FuncAnimation(fig, update, interval=1000, repeat=True)
        plt.show()
