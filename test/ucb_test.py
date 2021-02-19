import math
import random

import random


# draw: [float] -> int
# pick an index from the given list of floats proportionally
# to the size of the entry (i.e. normalize to a probability
# distribution and draw according to the probabilities).
def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex

        choiceIndex += 1


# distr: [float] -> (float)
# Normalize a list of floats to a probability distribution.  Gamma is an
# egalitarianism factor, which tempers the distribtuion toward being uniform as
# it grows from zero to one.
def distr(weights, gamma=0.0):
    theSum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)


# exp3: int, (int, int -> float), float -> generator
# perform the exp3 algorithm.
# numActions is the number of actions, indexed from 0
# rewards is a function (or callable) accepting as input the action and
# producing as output the reward for that action
# gamma is an egalitarianism factor
def exp3(numActions, reward, gamma, rewardMin=0, rewardMax=1):
    weights = [1.0] * numActions

    t = 0
    while True:
        probabilityDistribution = distr(weights, gamma)
        choice = draw(probabilityDistribution)
        theReward = reward(choice, t)
        scaledReward = (theReward - rewardMin) / (rewardMax - rewardMin)  # rewards scaled to 0,1

        estimatedReward = 1.0 * scaledReward / probabilityDistribution[choice]
        weights[choice] *= math.exp(estimatedReward * gamma / numActions)

        yield choice, theReward, estimatedReward, weights
        t = t + 1


def test():
    numActions = 10
    numRounds = 10000

    biases = [1.0 / k for k in range(2, 12)]
    rewardVector = [[1 if random.random() < bias else 0 for bias in biases] for _ in range(numRounds)]
    rewards = lambda choice, t: rewardVector[t][choice]
    # 计算最优的臂(娱乐、健康类目等)
    bestAction = max(range(numActions), key=lambda action: sum([rewardVector[t][action] for t in range(numRounds)]))
    bestUpperBoundEstimate = 2 * numRounds / 3
    gamma = math.sqrt(numActions * math.log(numActions) / ((math.e - 1) * bestUpperBoundEstimate))

    cumulativeReward = 0
    bestActionCumulativeReward = 0
    weakRegret = 0

    t = 0
    for (choice, reward, est, weights) in exp3(numActions, rewards, gamma):
        cumulativeReward += reward
        bestActionCumulativeReward += rewardVector[t][bestAction]
        weakRegret = (bestActionCumulativeReward - cumulativeReward)
        regretBound = (math.e - 1) * gamma * bestActionCumulativeReward + (numActions * math.log(numActions)) / gamma

        t += 1
        if t >= numRounds:
            break

    print(cumulativeReward)

if __name__ == '__main__':

    test()