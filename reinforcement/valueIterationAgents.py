# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
     
        #start at iteration zero
        currIterations=0
        
        #perform value iteration
        while currIterations < self.iterations:
            #initialize a counter to keep track of our values per iteration
            allVals = util.Counter()
            possStates=mdp.getStates()
            for state in possStates:
                #if there are no actions don't iterate, else compute
                if not mdp.isTerminal(state):
                    #initialize counter for getting values from the current state
                    vals = util.Counter()
                    possActions = mdp.getPossibleActions(state)
                    #iterate over actions and get their qvalues
                    for action in possActions:
                        vals[action] = self.computeQValueFromValues(state, action)
                    #get the best seen value for that state and action
                    allVals[state] = max(vals.values())
            currIterations += 1
            #update the policy with the best values
            self.values = allVals.copy()  


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        #get the Transition function and nextStates
        stateProbPairs=self.mdp.getTransitionStatesAndProbs(state,action)
        #initialize the value to zero
        actVal=0
        #iterate over probabilities (transition functions) and next states
        for pair in stateProbPairs:
            #compute qvalue
            actVal+=pair[1]*(self.mdp.getReward(state,action,pair[0])+self.discount*self.values[pair[0]])
        #print "The Q value is ",actVal
        return actVal
                        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        #return None if terminal
        if self.mdp.isTerminal(state):
            return None
        
        #get the legal actions
        actions=self.mdp.getPossibleActions(state)
        
        #if there are not legal actions return None
        if len(actions) == 0:
            return None
        
        #initialize a counter to hold our values
        values = util.Counter()
        #iterate over the legal actions and compute the qvalues
        for action in actions:
            values[action] = self.computeQValueFromValues(state, action)
        #return the best action
        return values.argMax()   

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
