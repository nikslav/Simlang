"""
Simple innate signalling simulation - communication in a population

pop_update takes a list of agents and picks two at random to be
producer and receiver for a random meaning. Each agent consists of 
a production system, a reception system and a list of 4 scores: the number of 
times they have successfully been understood as speaker, the number of times 
they have spoken, the number of times they have successfully understood as 
hearer, and the number of times they have been hearer, respectively

Usage example (copy and paste these commands at the prompt):

population = [[[[3, 1], [0, 2]], [[1,0], [2,4]], [0, 0, 0, 0]],
              [[[1, 0], [0, 1]], [[2,0], [0,1]], [0, 0, 0, 0]],
              [[[0, 1], [1, 0]], [[0,1], [1,0]], [0, 0, 0, 0]]]

for i in range(10000): pop_update(population)

print population

will do the following 10000 times: pick one of these three agents to be speaker 
and another to be hearer, have them communicate, and update their scores 
accordingly.

NOTE: there are two returns after the for loop in this usage example:
otherwise, Python will wait for you to add more code to the body of the for
loop.
"""

import random as rnd

def wta(items):
    maxweight = max(items)
    candidates = []
    for i in range(len(items)):
        if items[i] == maxweight:
            candidates.append(i)
    return rnd.choice(candidates)

def communicate(speaker_system, hearer_system, meaning):
    speaker_signal = wta(speaker_system[meaning])
    hearer_meaning = wta(hearer_system[speaker_signal])
    if meaning == hearer_meaning: 
        return 1
    else: 
        return 0

# ----- new code below -----

def pop_update(population):
    speaker_index = rnd.randrange(len(population))
    hearer_index = rnd.randrange(len(population) - 1) 
    if hearer_index >= speaker_index: hearer_index += 1     # ensure speaker
                                                  #and hearer are different
    speaker = population[speaker_index]
    hearer = population[hearer_index]
    meaning = rnd.randrange(len(speaker[0]))
    success = communicate(speaker[0], hearer[1], meaning)
    speaker[2][0] += success
    speaker[2][1] += 1
    hearer[2][2] += success
    hearer[2][3] += 1
