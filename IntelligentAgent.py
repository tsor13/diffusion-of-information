from agent import Agent
import numpy as np

class IntelligentAgent():

    def __init__(self, conn_index):
        self.conn_index = conn_index

    def MEU(self, conn_matrix, agents_decisions, util_matrix):
        """
        """
        utilities = []
        m, n = util_matrix.shape
        agent_count = len(conn_matrix[0, :])

        for decision_idx in range(m):
            utility = 0
            for i, conn_strength in enumerate(conn_matrix[self.conn_index, :]):
                other_agent_decision_idx = agents_decisions[i]
                utility += conn_strength * util_matrix[decision_idx, other_agent_decision_idx]

            utilities.append(utility)
        return np.argmax(utilities)




def test():
    conn_matrix = np.array([
        [0, 1, 1], #influence/conn on self is zero since undecided
        [2, 0, 3],
        [1, 1, 0]
    ])
    agents_decisions = np.array([
         0, #dem vote
        -1, #curr agent undecided -- doenst really matter what value this is
         1  #rep vote
    ])
    #util matrix doesnt need to take into account utility of other agent
    #and so only needs be a 2x2 matrix
    util_matrix = np.array([
        [2, 1],
        [0, 1]
    ])

    agent_1 = IntelligentAgent(1)

    assert agent_1.MEU(conn_matrix, agents_decisions, util_matrix) == 0
    print("SUCCESS: MEU")

# test()