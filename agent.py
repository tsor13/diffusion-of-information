import numpy as np, numpy.random

class Agent():
    """
    Represents a household in the cape town crisis that must decide which one of
    three water wells to choose to draw from

    variables:
    q_params - ndarray (3 x 3) - the parameters of the tri-noulli distribution
    for the likelihood of receiving an accurate signal.
    ex. --> An uninformed agent would have a q_matrix as follows:
    [[1/3, 1/3, 1/3],
     [1/3, 1/3, 1/3],
     [1/3, 1/3, 1/3]]

    ex. --> An informed agent might have a q matrix like the following, where
    the probability that the signal matches reality is higher
    [[1/4, 1/2, 1/4],
     [1/2, 1/4, 1/4],
     [1/4, 1/4, 1/2]]

    """

    def __init__(self, q_matrix):
        self.q = q_matrix

    def act_information_cascade(self, correct_well_index, observations):
        """
            returns highest probable good choice
        """
        private_info = np.random.choice([0,1,2], p = self.q[correct_well_index])

        # Add private info to observations
        observations = observations.copy()
        observations[private_info] += 1

        # Calculate probabilities based on private info and observations
        # This assumes equal probability of wells being active. It also discards
        # the marginal probability of the given info and observations, which is a constant for all 3 p values
        p = [ np.prod([self.q[i,j]**observations[j] for j in range(3)]) for i in range(3) ]
        print(observations)
        print(p)
        # Check for ties, choose randomly
        winning_indecies = p == np.max(p)
        choice = np.random.choice(np.where(winning_indecies)[0])
        print('choice:',choice)
        return choice
