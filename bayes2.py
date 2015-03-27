import random as rnd
from scipy.misc import logsumexp
from math import log, log1p, exp
from numpy import nan


learning = 'sample'     # The type of learning ('map' or 'sample')
bias = log(0.6)         # The preference for regular languages
variables = 2           # The number of different variables in the language
variants = 2            # The number of different variants each variable can take
noise = log(0.05)       # The probability of producing the wrong variant
population_size = 1000  # Size of population
teachers = 'single'     # Either 'single' or 'multiple' 
method = 'chain'        # Either 'chain' or 'replacement'

print learning
# ----- functions for dealing with log probabilities -----

def log_subtract(x,y):
    '''
    x-y in the log domain, i.e. equivalent to log(exp(x)-exp(y))
    '''
    return x + log1p(-exp(y-x))


def normalize_logprobs(logprobs):
    '''
    Same idea as normalize_probs, but in the log domain: takes a list of numbers 
    in the log domain, and normalises them, so that they sum to (0) (which is
    1 when converted to non-log probabilities). 
    '''
    logtotal=logsumexp(logprobs) #calculates the summed log probabilities
    normedlogs=[]
    for logp in logprobs:
        normedlogs.append(logp-logtotal) #normalise - subtracting in the log domain
                                        #equivalent to divising in the normal domain
    return normedlogs

def log_roulette_wheel(normedlogs):
    '''
    Selects a random index from normedlogs, with probability of selection being 
    proportional to probability. Note that this assumes that you feed in a normalised
    list of log probabilities.
    '''
    r=log(rnd.random()) #generate a random number in [0,1), then convert to log
    accumulator=normedlogs[0]
    for i in range(len(normedlogs)):
        if r<accumulator:
            return i
        accumulator=logsumexp([accumulator,normedlogs[i+1]])


def wta(items):
    '''
    This is good old winner-take-all - we are going to use it for MAP learners, 
    to pick the language with highest posterior probability.
    '''
    maxweight = max(items)
    candidates = []
    for i in range(len(items)):
        if items[i] == maxweight:
            candidates.append(i)
    return rnd.choice(candidates)

# ----- new code


def produce(language):
    '''
    Produces a variant for a particular language and randomly-selected variable.
    With log-probability given by the parameter noise, an incorrect randomly-
    selected variant is produced instead of the variant specified in language
    '''
    variable = rnd.randrange(len(language))
    correct_variant = language[variable]
    if log(rnd.random()) > noise:
        return [variable,correct_variant]
    else:
        possible_noise_variants = range(variants)
        possible_noise_variants.remove(correct_variant)
        noisy_variant = rnd.choice(possible_noise_variants)
        return [variable,noisy_variant]

def regular(language):
    '''
    Classifies a language as either regular (all variables expressed with the 
    same variant) or irregular (multiple variants used)
    '''
    regular = True
    first_variant = language[0]
    for variant in language:
        if variant != first_variant:
            regular = False
    return regular

def proportion_regular_language(population):
    '''
    A population is a list of languages, this just counts how many of them are 
    regular - used for outputting proportion of regular languages in our 
    iterated learning simulation.
    '''
    regular_count = 0
    for agent in population:
        if regular(agent):
            regular_count += 1
    return  regular_count / float(len(population))

def logprior(language):
    '''
    Calculates the log probability in the prior for a particular language. Note 
    that this must sum to log(1) for all languages, so there is some normalisation 
    in here.
    '''
    if regular(language):
        number_of_regular_languages = variants
        return bias - log(number_of_regular_languages) #subtracting logs = dividing
    else:
        number_of_irregular_languages = pow(variants, variables) - variants
        return log_subtract(0,bias) - log(number_of_irregular_languages)
        #log(1) is 0, so log_subtract(0,bias) is equivalent to (1-bias) in the 
        #non-log domain

def loglikelihood(data, language):
    '''
    Calculates the (log) likelihood of data given a language, which is simply 
    the product of the likelihoods of the individual data items.
    '''
    loglikelihoods = []
    logp_correct = log_subtract(0,noise) #probability of producing correct form
    logp_incorrect = noise - log((variants - 1)) #logprob of each incorrect variant
    for utterance in data:
        variable = utterance[0]
        variant = utterance[1]
        if variant == language[variable]:
            loglikelihoods.append(logp_correct)
        else:
            loglikelihoods.append(logp_incorrect)
    return sum(loglikelihoods) #summing log likelihoods = multiplying likelihoods

def all_languages(n):
    '''
    Generates a list of all possible languages for expressing n variables.
    '''
    if n == 0:
        return [[]]
    else:
        result = []
        smaller_langs = all_languages(n - 1)
        for l in smaller_langs:
            for v in range(variants):
                result.append(l + [v])
        return result

def learn(data):    
    '''
    Calculates the posterior probability for all languages, then picks a language.
    This will either be the maximum a posteriori language ('map')
    or a language sampled from the posterior.
    '''
    list_of_all_languages = all_languages(variables)
    list_of_posteriors = []
    for language in list_of_all_languages:
        this_language_posterior = loglikelihood(data,language) + logprior(language) 
        list_of_posteriors.append(this_language_posterior)
    if learning == 'map':
        map_language_index = wta(list_of_posteriors)
        map_language = list_of_all_languages[map_language_index]
        return map_language
    if learning == 'sample':
        normalized_posteriors = normalize_logprobs(list_of_posteriors)
        sampled_language_index = log_roulette_wheel(normalized_posteriors)
        sampled_language = list_of_all_languages[sampled_language_index]
        return sampled_language
        
def pop_learn(adult_population,bottleneck,number_of_learners):
    '''
    Generates a new population, consisting of a specified number_of_learners,
    who learn from data generated by the adult population - either from a single
    parent, or the whole population.
    '''
    new_population = []
    for n in range(number_of_learners):
        if teachers == 'single':
            potential_teachers = [rnd.choice(adult_population)]
        if teachers == 'multiple':
            potential_teachers = adult_population
        data = []
        for n in range(bottleneck):
            teacher = rnd.choice(potential_teachers)
            utterance = produce(teacher)
            data.append(utterance)
        learner_grammar = learn(data)
        new_population.append(learner_grammar)
    return new_population


def initial_population(n):
    '''
    Returns a list of n randomly-generated languages
    '''
    population = []
    possible_languages = all_languages(variables)
    for agent in range(n):
        language=rnd.choice(possible_languages)
        population.append(language)
    return population


def iterate(generations, bottleneck, report_every):
    '''
    Returns a list of two elements: final population, and accumulated data, 
    which is expressed in terms of proportion of the population using a regular 
    language
    '''
    population = initial_population(population_size)
    accumulator=[proportion_regular_language(population)]
    for g in range(1,generations+1):
        print '.',
        if method == 'chain': # Replace whole population
            population = pop_learn(population, bottleneck, population_size)
        if method == 'replacement': #Replace one individual at a time
            population = population[1:] 
            new_agent = pop_learn(population, bottleneck, 1)[0]
            population.append(new_agent)
        if (g % report_every == 0):
            accumulator.append(proportion_regular_language(population))
    return population,accumulator

