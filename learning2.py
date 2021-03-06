"""
Learned signalling with different weight-update rules

learn takes a signalling system, a meaning, a signal, and a learning rule of
the form [alpha, beta, gamma, delta] and adjusts the weights in that system
appropriately.

NB. alpha in the weight-update rule refers to the change made to connection
weights between active meaning and signal units, beta is the change made when the
meaning unit is active and the signal unit is inactive, gamma applies
when the meaning unit is inactive and the signal unit is active, and delta
applies when neither unit is active.

pop_learn uses a list of meaning-signal pairs to train a whole population of
learners. pop_produce uses a population to produce a list of meaning-signal
pairs

ca_monte_pop lets us test the communicative accuracy of the whole population

Usage example:

population = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
data = pop_produce(population, 100)
pop_learn(population, data, 100, [1, 0, 0, 0])
ca_monte_pop(population,10000)

This sequence of commands takes a population of two agents with all-0 weights, 
has them generate 100 meaning-signal pairs, then trains the same population on 
that data 100 times, using the frequency-counting learning rule from learning1.py, 
and then tests the communicative accuracy of the population after learning.
"""

import random as rnd

def production_weights(system, meaning):
    """
    Takes a system (a matrix or table of association weights) and retrieves 
    the weights relevant for producing for meaning. Since meanings are on the 
    rows, this will be the row indexed by meaning.
    """
    return system[meaning]

def reception_weights(system, signal):
    """
    Takes a system (a matrix or table of association weights) and retrieves 
    the weights relevant for receiving for signal. Since signals are on the 
    columns, this involves going through every row in the system, and 
    retrieving the column indexed by signal.
    """
    weights = []
    for row in system:
        weights.append(row[signal])
    return weights

def wta(items):
    """
    Returns the index (position) of the highest value in the list items; if 
    the highest value occurs multiple times, one of these is selected at random
    """
    maxweight = max(items)
    candidates = []
    for i in range(len(items)):
        if items[i] == maxweight:
            candidates.append(i)
    return rnd.choice(candidates)


def communicate(speaker_system, hearer_system, meaning):
    """
    Simulates a single communicative episode between speaker_system and 
    hearer_system about meaning, where speaker_system and hearer_system are 
    lists of lists representing matrices (i.e. tables) of production and 
    reception weights. Uses production_weights and reception_weights 
    to retrieve the relevant weights from those matrices, and wta to perform 
    production and reception.
    """
    speaker_signal = wta(production_weights(speaker_system,meaning))
    hearer_meaning = wta(reception_weights(hearer_system,speaker_signal))
    if meaning == hearer_meaning: 
        return 1
    else: 
        return 0

# ----- new code below -----


def learn(system, meaning, signal, rule):
    for m in range(len(system)):
        for s in range(len(system[m])):
            if m == meaning and s == signal: system[m][s] += rule[0]
            if m == meaning and s != signal: system[m][s] += rule[1]
            if m != meaning and s == signal: system[m][s] += rule[2]
            if m != meaning and s != signal: system[m][s] += rule[3]

def pop_learn(population, data, no_learning_episodes, rule):
    for n in range(no_learning_episodes):
        ms_pair = rnd.choice(data)
        learn(rnd.choice(population), ms_pair[0], ms_pair[1], rule)

def pop_produce(population, no_productions):
    ms_pairs = []
    for n in range(no_productions):
        speaker = rnd.choice(population)
        meaning = rnd.randrange(len(speaker))
        signal = wta(production_weights(speaker, meaning))
        ms_pairs.append([meaning,signal])
    return ms_pairs

def ca_monte_pop(population, trials):
    total = 0.
    for n in range(trials):
        speaker = rnd.choice(population)
        hearer = rnd.choice(population)
        total += communicate(speaker, hearer, rnd.randrange(len(speaker)))
    return total / trials
