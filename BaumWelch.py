

import numpy as np

# 3 parameters for the model:
# A  = transition probability matrix for hidden states Z

# A 
#                  to_state
#                 Z1 Z2 .. ZK
#              Z1 .. .. .. .. 
#              Z2 .. .. .. .. 
# from_state   .. .. .. .. .. 
#              ZK .. .. .. .. 
#             
#

# B  = emission probability matrix (prob of observing emission given we are in state)
# We say something like b_j(y_i) the probability of observing emission
# y_i given we are in state b_j
#
# B
#
#                    observation      
#                   
#                    Y1 Y2 .. Yi
#                Z1  .. .. .. ..
# hidden state   Z2  .. .. .. ..
#                ..  .. .. .. .. 
#                Zk  .. .. .. .. 
#
#
#
#

# pi = starting distribution

# alpha_i is prob of seeing observations {y_1,y_2,...,y_t} AND 
# being in state i at time $t$

# Forward procedure 

def forward(obs, A, B, pi):

    num_obs = len(obs)
    num_states= A.shape[0]

    alpha = np.zeros((num_obs, A.shape[0]))
    # so alpha is something like this, with T rows and num
    # cols equivalent to num hidden states
    #
    #        Z1 Z2 Z3 Z4
    #   y_1  .. .. .. ..
    #   y_2  .. .. .. .. 
    #   y_3  .. .. .. .. 
    #   ..   .. .. .. ..
    #   y_T  .. .. .. .. 
    #

    #initialise first row, element-wise multiplication for 
    # pi vector of being in each state
    # mult nby emission prob corresponding to each state
    alpha[0,:] = pi * B[:, V[0]]

    # now loop thru each element of alpha by row, col

    for row in range(num_states):
        for col in range(1, num_obs):
            alpha[col, row] = alpha[col - 1].dot(a[:, row]) * b[row, V[col]] 
    
    # now return alpha

    return alpha


def backward(obs, A, B):

    #similar as what we did for alpha...
    num_obs = len(obs)
    num_states= A.shape[0]

    beta = np.zeros((num_obs, A.shape[0]))

    # setting beta(T) = 1
    # set final col of beta = 1
    beta[num_obs - 1] = np.ones((num_states))

    #Now go recursively back thru t=T-1 to t=0
    #Except start at - 2 rather than - 1 (Python indexed from 0)

    for row in range(num_obs - 2, -1, -1):
        for col in range(num_states):
            # beta... = probability that we came from prev state
            # given current state / observation
            beta[row, col] = (beta[row + 1] * B[:, V[row + 1]]).dot(A[col, :])

    return beta


def baum_welch(obs, A, B, pi, iterations):
    Z = a.shape[0] #num hidden states
    T = len(obs) #num observations

    for n in range(iterations):
        alpha = forward(obs, A, B, pi)
        beta = backward(obs, A, B)

        xi = np.zeros((Z, Z, T - 1))

        for t in range(T - 1):
            denom = np.dot(np.dot(alpha[t, :].T, A) * B[:, obs[t + 1]].T, beta[t + 1, :])
            for i in range(Z):
                numerator = alpha[t, i] * A[i, :] * B[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denom

        gamma = np.sum(xi, axis=1)
        A = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = b.shape[1]
        denom = np.sum(gamma, axis=1)
        for l in range(K):
            B[:, l] = np.sum(gamma[:, V == l], axis=1)

        B = np.divide(B, denom.reshape((-1, 1)))

        if n%20==0:
            print(str(n) + ' iterations, estimates: ')
            print('A: ' + str(A))
            print('B: ' + str(B))

    return (A, B)

# Transition Probabilities
a = np.ones((2, 2))
a = a / np.sum(a, axis=1)

# Emission Probabilities
b = np.array(((1, 3, 5), (2, 4, 6)))
b = b / np.sum(b, axis=1).reshape((-1, 1))

# Equal Probabilities for the initial distribution
initial_distribution = np.array((0.5, 0.5))

#Usage:
#baum_welch(V, a, b, initial_distribution, 100)
