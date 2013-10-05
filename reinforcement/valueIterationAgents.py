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
        self.iterations = iterations=100
        self.values = util.Counter() # A Counter is a dict with default 0

        #Get all of the states
        
        #print "number of states is ",len(possStates)
        
        currIterations=0
        newValues=[]
        #print "current iterations: ",currIterations
        #print "max iterations: ",iterations
        while currIterations<self.iterations:
            #print "do I enter this while loop? ",currIterations
            possStates=mdp.getStates()
            for state in possStates:
                
                #print "this for loop? ",len(possStates)
                #If it is terminal, no future rewards, and you continue
                if mdp.isTerminal(state):
                    continue
                #Find all the actions for each state
                possActions=mdp.getPossibleActions(state)
                #Keep track of the best action, and it's value
                maxVal=-999999999
                for action in possActions:
                    actVal=self.computeQValueFromValues(state, action)
                    if actVal>maxVal:
                            maxVal=actVal
                newValues+=[(state,maxVal)]
                #print "Our new values are: ",newValues
            for val in newValues:
                #print "do we get here? ",len(newValues)
                self.values[val[0]]=val[1]
                print "Updated value is ",self.values[val[0]]
            currIterations+=1
            newValues=[]
        #print "can we break this while loop ever?"
                
                
            



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
        stateProbPairs=self.mdp.getTransitionStatesAndProbs(state,action)
        actVal=0
        for pair in stateProbPairs:
            actVal+=pair[1]*(self.mdp.getReward(state,action,pair[0])+self.discount*self.values[state])
        print "The Q value is ",actVal
        return actVal
                        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        
        actions=self.mdp.getPossibleActions(state)
        maxStateVal=-1000000
        bestAction=None
        for action in actions:
            nextState=self.mdp.getTransitionStatesAndProbs(state, action)
            if self.values[nextState[0]]>maxStateVal:
                #print "the value is: ",self.values[nextState[0]]
                #print "the current max is: ",maxStateVal
                maxStateVal=self.values[nextState[0]]
                bestAction=action
        print bestAction
        return bestAction
            
            
            

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
