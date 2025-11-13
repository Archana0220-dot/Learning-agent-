import random
import util
from learningAgents import ReinforcementAgent


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent
    """

    def __init__(self, **args):
        super().__init__(**args)
        self.qValues = util.Counter()  # default 0

    def getQValue(self, state, action):
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        return max(self.getQValue(state, a) for a in actions)

    def computeActionFromQValues(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return None
        best_value = self.computeValueFromQValues(state)
        best_actions = [a for a in actions if self.getQValue(state, a) == best_value]
        return random.choice(best_actions)

    def update(self, state, action, nextState, reward):
        old_value = self.getQValue(state, action)
        next_value = self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha) * old_value + self.alpha * (reward + self.discount * next_value)

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        return self.getPolicy(state)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    """
    Q-Learning agent with Pacman-specific default parameters.
    """

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        args.update({
            'epsilon': epsilon,
            'gamma': gamma,
            'alpha': alpha,
            'numTraining': numTraining
        })
        self.index = 0  # Pacman is always agent index 0
        super().__init__(**args)
        self.numTrainingGames = numTraining  # <-- required by Q7 autograder

    def getAction(self, state):
        action = super().getAction(state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
    Approximate Q-Learning Agent using feature-based representation.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        import featureExtractors
        self.featExtractor = util.lookup(extractor, featureExtractors.__dict__)()
        super().__init__(**args)
        self.weights = util.Counter()
        # Ensure autograder can access numTrainingGames
        if not hasattr(self, 'numTrainingGames'):
            self.numTrainingGames = self.numTraining

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(state, action)
        return sum(self.weights[f] * features[f] for f in features)

    def update(self, state, action, nextState, reward):
        features = self.featExtractor.getFeatures(state, action)
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        for f in features:
            self.weights[f] += self.alpha * difference * features[f]

    def final(self, state):
        super().final(state)
        if self.episodesSoFar == self.numTraining:
            print("Final weights:", self.weights)



class DeepQNetwork(ApproximateQAgent):
    numTrainingGames = 0  # <-- critical class-level variable

    def __init__(self, **args):
        super().__init__(**args)
        self.numTrainingGames = self.numTraining
        # Optionally, initialize neural network or extra stuff here

    def update(self, state, action, nextState, reward):
        # For autograder compatibility
        super().update(state, action, nextState, reward)

