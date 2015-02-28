import random as rnd
import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties
#import matplotlib.pyplot as plt
#import numpy as np

#fontP = FontProperties()
#fontP.set_size('small')

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
    

speakera = [[1,0,0],[0,0,1],[0,1,0]] # 11.11%
hearera = [[1,1,1],[0,1,0],[0,0,1]] 
speakerb = [[1,1,1],[1,1,1],[1,1,1]] # 33.33%
hearerb = [[1,1,1],[1,1,1],[1,1,1]]
hearerc = [[1,1],[1,1]]             # 50%
speakerc = [[1,1],[1,1]]
hearerd = [[1,1],[0,1]]             # 75%
speakerd = [[1,0],[0,1]]

linea = []
lineb = []
linec = []
lined = []
linee = [0 for i in range (10000)]
for i in range (10000):
    linea.append(0.111111),
    lineb.append(0.333333),
    linec.append(0.75),
    lined.append(0.50)
tock = []
for i in range (10001):
    if i%500 == 0:
        tock.append(i) 

plt.figure() #such important! wow!

results_a =[]
results_b =[]
results_c =[]
results_d =[]

for run in range (10):
    results_a.append(ca_monte(speakera,hearera, 10000))#, 'c:', label = "10 runs with predicted 11.11% accuracy" if r == 0 else '')
for run in range (10):
    results_b.append(ca_monte(speakerb,hearerb, 10000))#, 'm:', label = "10 runs with predicted 33.33% accuracy" if r == 0 else '')
for run in range (10):
    results_c.append(ca_monte(speakerc,hearerc, 10000))#, 'g:', label = "10 runs with predicted 50% accuracy" if r == 0 else '')
for run in range (10):
    results_d.append(ca_monte(speakerd,hearerd, 10000))#, 'y:', label = "10 runs with predicted 75% accuracy" if r == 0 else '')
   
diff_a =[[abs(element - 0.111111111111111) for element in list] for list in results_a]
diff_b =[[abs(element - 0.333333333333333) for element in list] for list in results_b]
diff_c =[[abs(element - 0.50) for element in list] for list in results_c]
diff_d =[[abs(element - 0.75) for element in list] for list in results_d]


for run in range (10):
    plt.plot(results_a[run], 'c:', label = "10 runs with predicted 11.11% accuracy" if run == 0 else '')
for run in range (10):
    plt.plot(results_b[run], 'm:', label = "10 runs with predicted 33.33% accuracy" if run == 0 else '')
for run in range (10):
    plt.plot(results_c[run], 'g:', label = "10 runs with predicted 50% accuracy" if run == 0 else '')
for run in range (10):
    plt.plot(results_d[run], 'y:', label = "10 runs with predicted 75% accuracy" if run == 0 else '')
   

plt.plot(linea, 'k', label = "Predicted accuracy")
plt.plot(lineb, 'k', )
plt.plot(linec, 'k', )
plt.plot(lined, 'k')




#plt.legend (loc = 1, prop = fontP)

plt.ylabel ("Communicative Success Rate")
plt.xlabel ("Number of iterations")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3)

#plt.savefig("d:/my_figure.pdf")
plt.show()

plt.figure()

for run in range (1):
    plt.plot(diff_e[run], 'c:', label = "" if run == 0 else '')
for run in range (1):
    plt.plot(diff_b[run], 'm:', label = "" if run == 0 else '')
for run in range (1):
    plt.plot(diff_c[run], 'g:', label = "" if run == 0 else '')
for run in range (1):
    plt.plot(diff_d[run], 'y:', label = "" if run == 0 else '')
    
plt.show()