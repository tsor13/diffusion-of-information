from graph import Graph
from agent import Agent
import numpy as np

def prob_1():
    # Setup
    # 1-cycle
    matrix1 = np.array([[0,1,0,0,0,0,1],
                        [1,0,1,0,0,0,0],
                        [0,1,0,1,0,0,0],
                        [0,0,1,0,1,0,0],
                        [0,0,0,1,0,1,0],
                        [0,0,0,0,1,0,1],
                        [1,0,0,0,0,1,0]])
    # All agents start with a 50/50 chance of having correct information
    q = np.array(  [[.5,.25,.25],
                    [.25,.5,.25],
                    [.25,.25,.5]])

    # Initialize the Agents
    nodes = [Agent(q) for _ in range(7)]

    # Create the graph
    one_cycle_graph = Graph(matrix1, nodes)

    one_cycle_graph.run_day()
