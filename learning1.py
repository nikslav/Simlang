"""
Simple learned signalling system simulation

learn takes a three arguments:
1. a signalling system (represented as a single matrix of
association weights, with meanings on rows, signals on columns),
2. a meaning (an integer),
3. a signal (also an integer)
and increases the association weight for that meaning-signal pair. 

train does the same but for a list of meaning-signal pairs.

Usage example:

system = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
train(system, [[0, 0], [1, 1], [2, 2]])

This simulates a learner being exposed to three meaing-signal pairs:
    meaning 1 paired with signal 1
    meaning 2 paired with signal 2
    meaning 3 paired with signal 3
ca_monte can be used as before to test whether an agent that has
learned a particular signalling system can talk to another one.
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


def ca_monte(speaker_system, hearer_system, trials):
    """
    Simlates trials communicative events between speaker_system and hearer_system,
    and returns a list of length trials, which is a trial-by-trial record of the
    proportion of succesful communications. Uses the communicate function to 
    simulate each communicatiev event, and selects a random meaning for every
    communication event.
    """
    total = 0.
    accumulator = []
    for n in range(trials):
        total += communicate(speaker_system, hearer_system,
                             rnd.randrange(len(speaker_system)))
        accumulator.append(total/(n+1))
    return accumulator

# ----- new code for learning below -----

def learn(system, meaning, signal):
    system[meaning][signal] += 1

def train(system, ms_pair_list):
    for ms_pair in ms_pair_list:
        learn(system, ms_pair[0], ms_pair[1])

