import random as rnd
 

def weighted_random (items):
    listsum = sum(items)
    treshold = rnd.random()*listsum
    accumulator = 0
    for i in range(len(items)):
        accumulator += items[i]
        if accumulator >= treshold:
            return i,items[i]
            
def sq_wr (items):
    items_sum = sum (element ** 2 for element in items)
    treshold = rnd.random()*items_sum 
    accumulator = 0
    for i in range (len(items)):
	accumulator += items[i]**2
	if accumulator >=treshold:
		return items[i]