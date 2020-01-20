import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.001
        self.alpha = 1.0
        self.gamma = 1.0
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        probs = np.ones(self.nA) * self.epsilon / self.nA
        probs[np.argmax(self.Q[state])] = 1-self.epsilon + (self.epsilon / self.nA)   
        action = np.random.choice(np.arange(self.nA), p=probs)
        
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            probs = np.ones(self.nA) * self.epsilon / self.nA
            probs[np.argmax(self.Q[next_state])] = 1-self.epsilon + (self.epsilon / self.nA)   
                
            expcted_reward  = np.sum(np.multiply(probs,self.Q[next_state]))
            self.Q[state][action]+=self.alpha*(reward+self.gamma*expcted_reward-self.Q[state][action])
            
        