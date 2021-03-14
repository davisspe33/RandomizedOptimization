import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import sklearn
import matplotlib.pyplot as plt 
import datetime

edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),(1, 5),(5, 4),(3, 5),(7,6),(6,5),(7,1),(2,3),(8,2),(8,4),(8,6),(9,7),(9,5),(1,7)]
fitness = mlrose.MaxKColor(edges)
problem = mlrose.DiscreteOpt(length = 10, fitness_fn = fitness, maximize = True, max_val=2)

max_iters=1000


start = datetime.datetime.now()
best_statea, best_featurea,curvea= mlrose.random_hill_climb(problem, max_attempts=100, max_iters=max_iters, restarts=0, init_state=None, curve=True, random_state=None)
stop = datetime.datetime.now()
print('Execution Time (milli) random_hill_climb: ',((stop - start).microseconds)/1000)

start = datetime.datetime.now()
best_stateb, best_featureb,curveb = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=max_iters, curve=True, random_state=None)
stop = datetime.datetime.now()
print('Execution Time (milli) genetic_alg: ',((stop - start).microseconds)/1000)

start = datetime.datetime.now()
best_statec, best_featurec,curvec = mlrose.simulated_annealing(problem, max_attempts = 10, max_iters = max_iters, curve=True, init_state = None, random_state = 1)
stop = datetime.datetime.now()
print('Execution Time (milli) simulated_annealing: ',((stop - start).microseconds)/1000)

start = datetime.datetime.now()
best_stated, best_featured,curved= mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=100, max_iters=max_iters, curve=True, random_state=None, fast_mimic=True)
stop = datetime.datetime.now()
print('Execution Time (milli) mimic: ',((stop - start).microseconds)/1000)

param_rangea=np.linspace(1, 1000,curvea.shape[0])
param_rangeb=np.linspace(1, 1000,curveb.shape[0])
param_rangec=np.linspace(1, 1000,curvec.shape[0])
param_ranged=np.linspace(1, 1000,curved.shape[0])

plt.grid()
plt.plot(param_rangea, curvea, label="random_hill_climb",color="darkorange", lw=2)
plt.plot(param_rangeb, curveb, label="genetic_alg",color="blue", lw=2)
plt.plot(param_rangec, curvec, label="simulated_annealing",color="green", lw=2)
plt.plot(param_ranged, curved, label="mimic",color="red", lw=2)
plt.legend(loc="best")
plt.title("MaxKColor")
plt.xlabel("Iterations")
plt.ylabel("fitness")
plt.show()

print('done')