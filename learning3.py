"""
Cultural evolution by iterated learning

simulation creates an initial population of agents with all-zero connection
weights and uses this population to produce a data set of utterances.
These utterances are used to train the population at the next time step.
Different methods to update the population are included:

replacement - this implements the 'replacement method' whereby the oldest
agent is replaced by a new one each generation

chain - this implements a 'transmission chain' in which, at each time step the
whole population is replaced

closed - this implements the 'closed group' method, in which the population is
static and no new individuals are ever added

You can also specify what kind of language the initial population uses: 'optimal'
means that the intitial population has their weights set such that they will 
communicate optimally; 'random' (or in fact anything other than 'optimal') means
that the initial population has all-0 connection weights and will produce random 
meaning-signal pairs.

There are a number of other global simulation parameters for simulation, commented
below.

Usage example:

final_pop,ca = simulation(500, 1000, 10)

This runs the simulation for 500 'generations', with 1000 trials for the monte
carlo calculation, and returns a list of two elements: the final population,
and a list of 50 values, indicating communicative accuracy (as evaluated by
monte carlo simulations) for every tenth generation. This data can be plotted
in the usual way.

""" 

import random as rnd
import matplotlib as plt

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
    weights = [row[signal] for row in system]
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


def learn(system, meaning, signal, rule):
    """
    Simulates a single learning episode. The learner with weight matrix system 
    learns from the observation of meaning and signal, updating their weight
    matrix according to rule. The rule is of the form [alpha,beta,gamma,delta], 
    where the weight-change specified in alpha applies to connections between 
    active meaning and signal nodes, beta applies where the meaning is active but
    the signal is inactive, gamma applies where the meaning is inactive but
    the signal is active, and delta applies where both are inactive.
    """
    for m in range(len(system)):
        for s in range(len(system[m])):
            if m == meaning and s == signal: system[m][s] += rule[0]
            if m == meaning and s != signal: system[m][s] += rule[1]
            if m != meaning and s == signal: system[m][s] += rule[2]
            if m != meaning and s != signal: system[m][s] += rule[3]

def pop_learn_each(population, data, no_learning_episodes, rule):
    """
    Trains a population (a list of agents) on meaning-signal pairs selected 
    randomly from data. Data is a list of meaning-signal pairs, where each item
    in data is a list of length 2: [meaning,signal]. no_learning_episodes are 
    simulated; at each episode, a learner is selected randomly from population, 
    a meaning-signal pair is selected randomly from data, then the learner
    learns fromn that observation according to rule.
    """
    for learner in (population):
        for n in range(no_learning_episodes):
            ms_pair = rnd.choice(data)
            learn(learner, ms_pair[0], ms_pair[1], rule)
            
def pop_learn (population, data, no_learning_episodes, rule):
    for n in range(no_learning_episodes):
        ms_pair = rnd.choice(data)
        learn(rnd.choice(population), ms_pair[0], ms_pair[1], rule)

def pop_produce(population, no_productions):
    """
    Generates data - a list of meaning-signal pairs, where each is of the form
    [meaning,signal] - from population. For each production (up to no_productions), 
    a random agent is selected from population, a random meaning is selected, 
    and then the chosen agent produces for that meaning according to the usual 
    wta procedure.
    """
    ms_pairs = []
    for n in range(no_productions):
        speaker = rnd.choice(population)
        meaning = rnd.randrange(len(speaker))
        signal = wta(production_weights(speaker, meaning))
        ms_pairs.append([meaning,signal])
    return ms_pairs

def ca_monte_pop(population, trials):
    """
    Evaluates communicative accuracy within population, using trials episodes
    of communication (featuring a random speaker and hearer, who may be the same
    individual). Returns proportion of trials on which communication was succesful.
    """
    total = 0.
    for n in range(trials):
        speaker = rnd.choice(population)
        hearer = rnd.choice(population)
        total += communicate(speaker, hearer, rnd.randrange(len(speaker)))
    return total / trials

# ----- new code below -----

meanings = 10            # number of meanings
signals = 10             # number of signals
interactions_talk = 100 # number of utterances produced and the number
interactions = 100        # of times this set is randomly sampled for training.
size = 100              # size of population
method = 'replacement'  # method of population update
initial_language_type = 'random' # either 'optimal' or 'random'
rule = [1, 0, 0, 1]     # learning rule (alpha, beta, gamma, delta)

def new_agent(initial_language_type):
    system = []
    for row_n in range(meanings):
        row = []
        for column_n in range(signals):
            if (initial_language_type=='optimal') & (row_n==column_n):
                row.append(1)
            else:             
                row.append(0)
        system.append(row)
    return system

def new_population(size,initial_language_type):
    population = []
    for i in range(size):
        population.append(new_agent(initial_language_type))
    return population

def simulation(generations, mc_trials, report_every, method=method):
    population = new_population(size,initial_language_type)
    data_accumulator=[]
    for i in range(generations+1):
        #print '.', #comment this line out if you don't want the dots
        if (i % report_every == 0):
            print i
            data_accumulator.append(ca_monte_pop(population, mc_trials))
        data = pop_produce(population, interactions_talk)
        if method == 'chain': 
            population = new_population(size, 'random')
            pop_learn(population, data, interactions, rule)
        if method == 'replacement':
            population = population[1:] #This removes the first item of the list
            learner=new_agent('random')
            pop_learn([learner], data, interactions, rule)
            population.append(learner)
        if method == 'closed':
            pop_learn(population, data, interactions, rule)
    return [population,data_accumulator]

"""
rule = [1,0,0,1]
chain_sim_const_500_1000_10 = []
repl_sim_const_500_1000_10 = []
closed_sim_const_500_1000_10 = []
chain_sim_const_int_x_100_50_1000_1=[]
closed_sim_const_50_1000_1 = []

for i in range (100):
    repl_sim_const_500_1000_10.append(simulation(500,1000, 10, method = 'replacement')) 

for i in range (100):
    chain_sim_const_500_1000_10.append(simulation(500,1000,10)) 
    
for i in range (100):
    closed_sim_const_500_1000_10.append(simulation(500,1000, 10, method = 'closed')) 

for i in range (100):
    closed_sim_const_50_1000_1.append(simulation(50,1000, 1, method = 'closed'))   
    
rule = [1, 0, 0, 0]
chain_sim_maint_500_1000_10 = []
repl_sim_maint_500_1000_10 = []
closed_sim_maint_500_1000_10 = []
chain_sim_maint_int_x_100_50_1000_1=[]
closed_sim_maint_50_1000_1 = []

for i in range (100):
    repl_sim_maint_500_1000_10.append(simulation(500,1000, 10, method = 'replacement')) 

for i in range (100):
    chain_sim_maint_500_1000_10.append(simulation(500,1000,10)) 
    
for i in range (100):
    closed_sim_maint_500_1000_10.append(simulation(500,1000, 10, method = 'closed')) 

for i in range (100):
    closed_sim_maint_50_1000_1.append(simulation(50,1000, 1, method = 'closed')) 


interaction = 10000
for i in range (100):
    chain_sim_const_int_x_100_50_1000_1.append((simulation(50,1000,1))) 

for i in range (100):
    chain_sim_maint_int_x_100_50_1000_1.append((simulation(50,1000,1))) 
"""

rule = [1,0,0,1]
chain_sim_const_1000_1000_20 = []
repl_sim_const_1000_1000_20 = []
closed_sim_const_1000_1000_20 = []


for i in range (100):
    repl_sim_const_1000_1000_20.append(simulation(1000,1000, 20, method = 'replacement')) 
print '\a'
for i in range (100):
    chain_sim_const_1000_1000_20.append(simulation(1000,1000,20)) 
print '\a'    
for i in range (100):
    closed_sim_const_1000_1000_20.append(simulation(1000,1000, 20, method = 'closed')) 
print '\a'

rule = [1, 0, 0, 0]

chain_sim_maint_1000_1000_20 = []
repl_sim_maint_1000_1000_20 = []
closed_sim_maint_1000_1000_20 = []

for i in range (100):
    repl_sim_maint_1000_1000_20.append(simulation(1000,1000, 20, method = 'replacement')) 
print '\a'
for i in range (100):
    chain_sim_maint_1000_1000_20.append(simulation(1000,1000,20)) 
print '\a'   
for i in range (100):
    closed_sim_maint_1000_1000_20.append(simulation(1000,1000, 20, method = 'closed')) 
print '\a'
    
interaction = 10000
chain_sim_maint_int_x_100_1000_1000_20 =[]
chain_sim_const_int_x_100_1000_1000_20 =[]
closed_sim_maint_int_x_100_100_1000_20=[]
closed_sim_const_int_x_100_100_1000_20=[]

rule = [1,0,0,1]
for i in range (100):
    closed_sim_const_int_x_100_100_1000_20.append(simulation(1000,1000, 20, method = 'closed'))

for i in range (100):
    chain_sim_const_int_x_100_1000_1000_20.append((simulation(1000,1000,20))) 

rule = [1,0,0,0]
for i in range (100):
    closed_sim_maint_int_x_100_100_1000_20.append(simulation(1000,1000, 20, method = 'closed')) 

for i in range (100):
    chain_sim_maint_int_x_100_1000_1000_20.append((simulation(1000,1000,20)))

 

    
f = open( 'd:\sim.py', 'w' )
f.write(repr(chain_sim_const_1000_1000_20) + '\n' )
f.write(repr(repl_sim_const_1000_1000_20) + '\n' )
f.write(repr(closed_sim_const_1000_1000_20) + '\n' )
f.write(repr(chain_sim_const_int_x_100_1000_1000_20) + '\n' )
f.write(repr(closed_sim_const_int_x_100_100_1000_20) + '\n' )
f.write(repr(closed_sim_maint_int_x_100_100_1000_20) + '\n' )
f.write(repr(chain_sim_maint_1000_1000_20) + '\n' )
f.write(repr(repl_sim_maint_1000_1000_20) + '\n' )
f.write(repr(closed_sim_maint_1000_1000_20) + '\n' )
f.write(repr(chain_sim_maint_int_x_100_1000_1000_20) + '\n' )
f.write(repr(chain_sim_maint_1000_1000_20) + '\n' )
f.write(repr(repl_sim_const_1000_1000_20) + '\n' )
f.close()
print '\a'

"""for i in range(40):
    plot(chain_sim_const_1000_1000_20[i][1],'g')
    plot(repl_sim_const_1000_1000_20[i][1], 'b' )
    plot(closed_sim_const_1000_1000_20[i][1],'r') 
    plot(chain_sim_maint_int_x_100_1000_1000_20[i][1],'c')
    plot(closed_sim_const_int_x_100_100_1000_2[i][1],'m') """
    
"""for i in range(40):
    plot(chain_sim_maint_1000_1000_20[i][1],'g')
    plot(repl_sim_maint_1000_1000_20[i][1], 'b' )
    plot(closed_sim_maint_1000_1000_20[i][1],'r') 
    plot(chain_sim_maint_int_x_100_100_1000_2[i][1],'c')
    plot(chain_sim_maint_int_x_100_1000_1000_20[i][1],'m') """
    