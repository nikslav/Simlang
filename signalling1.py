"""
Simple innate signalling simulation

ca_monte returns communicative accuracy between a speaker (or sender, or 
producer) system and a hearer (or receiver) system using monte carlo simulation.

Systems are expressed as a list of lists of association weights. Matrix rows in the
speaker system are meanings, columns are signals. In the hearer system, matrix
rows are signals, columns are meanings.

Production and reception are winner-take-all.

Usage example (note: I have presented the speaker and hearer systems row-by-row
to make the matrix structure clearer: there is no need to do this when using
the code, unless you feel it helps. This code should run if you just copy and 
paste it at the prompt).

In [2]: a_speaker_system = [[1, 0, 0],
   ...:                     [0, 1, 0],
   ...:                     [0, 1, 1]]

In [3]: a_hearer_system =  [[1, 0, 0],
   ...:                     [0, 1, 1],
   ...:                     [0, 0, 1]]

In [4]: ca_monte(a_speaker_system, a_hearer_system, 100)


Returns a list of expected communicative success values, in a trial-by-trial
list (so the first element in the list gives the proportion of successful
communications after 1 trial, the second gives the proportion of successful
events after two trials etc), based on 100 evaluations of communication between
the specified speaker and hearer systems.  There are three meanings
and three signals, but the communication system as specified above contains
some homonymy (the second signal can be used for either the second or third
meaning) and synonymy (the third meaning can be expressed using either the
second or third signal).
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

def ca_monte(speaker_system, hearer_system, trials):
    total = 0.
    accumulator = []
    for n in range(trials):
        total += communicate(speaker_system, hearer_system,
                             rnd.randrange(len(speaker_system)))
        accumulator.append(total/(n+1))
    return accumulator
