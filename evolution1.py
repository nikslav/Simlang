"""
Simple innate signalling system simulation - evolving population

simulation creates an initial population of random agents and evolves this
population over a number of generations, building a list of total fitness at
each generation. Fitness is calculated after a certain number of random
interactions among the population and is determined by the proportion of
successful 'sends' and successful 'receives', each scaled by a weighting factor.
Parents are selected with a probability proportional to their fitness and
there is a chance of mutation of each weight in each agent's speaking and 
hearing matrixes.

simulation returns a list of two elements: the final population, and the
list of generation-by-generation fitness scores. 

Usage example (you can copy and paste this to the prompt): 

my_pop,fitness_list = simulation(100)

This will run the simulation for 100 generations. It uses a rather clever way to 
assign the values returned by simulation to variables - we create two variables, 
my_pop and fitness_list. The first item returned by simulation, the final 
population,is stored in the variable my_pop. The second item, the list of
generation-by-generation fitness scores, is stored in fitness_list. 
If you want to see the final population, type my_pop at the prompt; if you want 
to see the fitness of the population over generations, type fitness_list at the 
prompt.
"""

import random as rnd


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
    reception weights. Uses wta to perform production and reception.
    """
    speaker_signal = wta(speaker_system[meaning])
    hearer_meaning = wta(hearer_system[speaker_signal])
    if meaning == hearer_meaning: 
        return 1
    else: 
        return 0

def pop_update(population):
    """
    Selects a random speaker and hearer from population, simulates a single 
    communicative interaction about a randomly-selected meaning using the 
    commnicate function, then updates the success scores for speaker and hearer.
    population is a list of agents. Each agent is a list of length three: the 
    first item in this list is that agent's production matrix, the second is the
    reception matrix, and the third is a list of counts (which is itself a list: 
    number of succesful sends, number of sends attempted, number of succesful 
    receives, number of receives attempted).
    """
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


# ----- new code below -----

from copy import deepcopy

"""
The following values are the parameters for the simulation - they determine
various features of how the model works. Often you will want to tinker with 
these, then re-run a simulation to see what happens. If you want to do that, 
then you have to edit the parameter values here, then re-run the code using the
play button  - we just found out in today's lab that, in Canopy, typing a new 
value at the prompt doesn't change anything, sorry!
Unless you have programmed in python before, doing anything fancier than this
like writing functions which edit the parameters, will get you in trouble: 
these paremeters are known as global variables, and altering them from within
a function requires some special syntax we won't cover on this course.
"""
mutation_rate = 0.001   # probability of mutation per weight
mutation_max = 1       # maximum value of a random weight
send_weighting = 10    # weighting factor for send score
receive_weighting = 10 # weighting factor for receive score
meanings = 3           # number of meanings
signals = 3            # number of signals
interactions = 1000    # number of interactions per generation
size = 100             # size of population

def fitness(agent):
    send_success = agent[2][0] 
    send_n = agent[2][1]       
    receive_success = agent[2][2]
    receive_n = agent[2][3]
    if send_n == 0: 
        send_n = 1
    if receive_n == 0: 
        receive_n = 1
    return ((send_success/send_n) * send_weighting +
            (receive_success/receive_n) * receive_weighting) + 1

def sum_fitness(population):
    total = 0
    for agent in population:
        total += fitness(agent)
    return total
    
def mutate(system):
    for row_i in range(len(system)):
        for column_i in range(len(system[0])):
            if rnd.random() < mutation_rate:
                system[row_i][column_i] = rnd.randint(0, mutation_max)

def pick_parent(population,sum_f):
    accumulator = 0
    r = rnd.uniform(0, sum_f)
    for agent in population:
        accumulator += fitness(agent)
        if r < accumulator:
            return agent

def new_population(population):
    new_p = []
    sum_f = sum_fitness(population)
    for i in range(len(population)):
        parent=pick_parent(population, sum_f)
        child_production_system = deepcopy(parent[0])
        child_reception_system = deepcopy(parent[1])
        mutate(child_production_system)
        mutate(child_reception_system)
        child=[child_production_system,
               child_reception_system,
               [0., 0., 0., 0.]]
        new_p.append(child)
    return new_p

def random_system(rows,columns):
    system = []
    for i in range(rows):
        row = []
        for j in range(columns):
            row.append(rnd.randint(0, mutation_max))
        system.append(row)
    return system

def random_population(size):
    population = []
    for i in range(size):
        population.append([random_system(meanings,signals),
                           random_system(signals,meanings),
                           [0., 0., 0., 0.]])
    return population

def simulation(generations):
    accumulator=[]
    population = random_population(size)
    for i in range(generations):
        for j in range(interactions):
            pop_update(population)
        average_fitness=(sum_fitness(population)/size)
        accumulator.append(average_fitness)
        print '.', #this prints out a dot every generation
                   #if this annoys you, comment this line out with a #
        #print i, average_fitness #uncomment this line if you would like updates 
                                  #on average fitness during runs
        population = new_population(population)
    return [population,accumulator]

