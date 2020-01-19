import random
import time
import ray

def knapsack_dp(W, wt, val, n): 
    T = [[0 for x in range(W+1)] for x in range(n+1)] 

    for i in range(n+1): 
        for w in range(W+1): 
            if i==0 or w==0: 
                T[i][w] = 0
            elif wt[i-1] <= w: 
                T[i][w] = max(val[i-1] + T[i-1][w-wt[i-1]],  T[i-1][w]) 
            else: 
                T[i][w] = T[i-1][w] 
  
    #print(T[n][W])

def func(x):
  return x[0]

def crossBreeding(pop,reserve_pop,fitness,reserve_fitness):
  breed_pop1=[]
  breed_pop2=[]
  for i in range(len(pop)):
    breed_pop1.append((fitness[i],pop[i]))
    breed_pop2.append((reserve_fitness[i],reserve_pop[i]))

  breed_pop1.sort(key=func,reverse=True)
  breed_pop2.sort(key=func,reverse=True)
  j,k=0,0
  for i in range(0,len(pop)):
    if breed_pop1[j][0]>breed_pop2[k][0]:
      pop[i]=breed_pop1[j][1]
      j+=1
    else:
      pop[i]=breed_pop2[k][1]
      k+=1


  return pop



def knapsack(weight, value, MAX, popSize, mut, maxGen, percent):
  generation = 0
  pop = generate(value, popSize)
  reserve_pop=generate(value,popSize)
  #fitness = getFitness(pop, weight, value, MAX)
  ray.init()
  fitness_id=getFitness.remote(pop,weight,value,MAX)
  fitness=ray.get(fitness_id)
  ray.shutdown()
  reserve_fitness=getFitness1(reserve_pop,weight,value,MAX)
  while(not test(fitness, percent) and generation < maxGen):
    generation += 1
    temp=pop.copy()
    #ray.init(ignore_reinit_error=True)
    pop = newPopulation(pop, fitness, mut)
    reserve_pop=newPopulation(reserve_pop,reserve_fitness,mut)
    #ret_id1 = newPopulation.remote(pop,fitness,mut)
    #ret_id2 = newPopulation.remote(reserve_pop,reserve_fitness,mut)
    #pop,reserve_pop = ray.get([ret_id1, ret_id2])
    #fitness = getFitness(pop, weight, value, MAX)
    ray.init()
    fitness_id=getFitness.remote(pop,weight,value,MAX)
    fitness=ray.get(fitness_id)
    ray.shutdown()
    reserve_fitness=getFitness1(reserve_pop,weight,value,MAX)
    pop=crossBreeding(pop,reserve_pop,fitness,reserve_fitness)
    reserve_pop=temp.copy()
    #fitness = getFitness(pop, weight, value, MAX)
    ray.init()
    fitness_id=getFitness.remote(pop,weight,value,MAX)
    fitness=ray.get(fitness_id)
    ray.shutdown()
 
  arr=selectElite(pop, fitness)
  return arr

def generate(value, popSize):
  length = len(value)
  pop = [[random.randint(0,1) for i in range(length)] for j in range(popSize)]
  return pop

@ray.remote
def getFitness(pop, weight, value, MAX):
  fitness = []
  for i in range(len(pop)):
    sum_weight = MAX+1
    sum_value = 0
    while (sum_weight > MAX):
      sum_weight = 0
      sum_value = 0
      ones = []
      for j in range(len(pop[i])):
        if pop[i][j] == 1:
          sum_weight += weight[j]
          sum_value += value[j]
          ones += [j]
      if sum_weight > MAX:
        pop[i][ones[random.randint(0, len(ones)-1)]] = 0
    fitness += [sum_weight]
  return fitness

def getFitness1(pop, weight, value, MAX):
  fitness = []
  for i in range(len(pop)):
    sum_weight = MAX+1
    sum_value = 0
    while (sum_weight > MAX):
      sum_weight = 0
      sum_value = 0
      ones = []
      for j in range(len(pop[i])):
        if pop[i][j] == 1:
          sum_weight += weight[j]
          sum_value += value[j]
          ones += [j]
      if sum_weight > MAX:
        pop[i][ones[random.randint(0, len(ones)-1)]] = 0
    fitness += [sum_weight]
  return fitness

def newPopulation(pop, fit, mut):
  popSize = len(pop)
  newPop = []
  newPop += [selectElite(pop, fit)]
  while(len(newPop) < popSize):
    (mate1, mate2) = select(pop, fit)
    newPop += [mutate(crossover(mate1, mate2), mut)]
  return newPop
  
def selectElite(pop, fit):

  elite = 0
  for i in range(len(fit)):
    if fit[i] > fit[elite]:
      elite = i
  return pop[elite]

def select(pop, fit):
  size = len(pop)
  totalFit = sum(fit)
  lucky = random.randint(0, totalFit)
  tempSum = 0
  mate1 = []
  fit1 = 0
  for i in range(size):
    tempSum += fit[i]
    if tempSum >= lucky:
      mate1 = pop.pop(i)
      fit1 = fit.pop(i)
      break
  tempSum = 0
  lucky = random.randint(0, sum(fit))
  for i in range(len(pop)):
    tempSum += fit[i]
    if tempSum >= lucky:
      mate2 = pop[i]
      pop += [mate1]
      fit += [fit1]
      return (mate1, mate2)

def crossover(mate1, mate2):
  lucky = random.randint(0, len(mate1)-1)
  return mate1[:lucky]+mate2[lucky:]
  
def mutate(gene, mutate):
  for i in range(len(gene)):
    lucky = random.randint(1, mutate)
    if lucky == 1:
      gene[i] = bool(gene[i])^1
  return gene
    
def test(fit, rate):
  maxCount = mode(fit)
  if float(maxCount)/float(len(fit)) >= rate:
    return True
  else:
    return False

def mode(fit):
  values = set(fit)
  maxCount = 0
  for i in values:
    if maxCount < fit.count(i):
      maxCount = fit.count(i)
  return maxCount

def profit(arr,value):
  sum=0
  for i in range(0,len(arr)):
    if arr[i]==1:
      sum+=value[i]
  return sum

'''weight = [random.randint(5,20) for i in range(200)]
value = [random.randint(5,100) for i in range(0,200)]
print(weight)
print(value)'''

weight=[20, 15, 16, 13, 11, 11, 16, 16, 10, 7, 11, 9, 17, 15, 15, 11, 13, 11, 14, 13, 20, 18, 6, 9, 8, 5, 14, 18, 19, 15, 5, 5, 12, 6, 17, 12, 20, 14, 17, 9, 17, 5, 11, 17, 14, 18, 15, 16, 8, 20, 13, 19, 15, 15, 18, 15, 12, 15, 11, 17, 13, 19, 15, 17, 10, 13, 10, 7, 14, 14, 12, 5, 8, 18, 12, 8, 10, 7, 9, 18, 7, 13, 15, 6, 13, 7, 11, 16, 6, 6, 10, 8, 18, 7, 13, 12, 9, 19, 18, 20, 19, 12, 7, 15, 12, 16, 20, 17, 12, 19, 7, 20, 6, 9, 10, 20, 16, 9, 10, 19, 6, 16, 13, 7, 8, 17, 10, 11, 7, 18, 5, 14, 16, 18, 15, 8, 11, 11, 18, 7, 6, 6, 18, 14, 18, 5, 10, 20, 20, 5, 14, 9, 19, 17, 17, 9, 16, 18, 11, 20, 15, 6, 9, 15, 18, 16, 8, 6, 19, 8, 10, 18, 20, 7, 18, 20, 15, 5, 17, 9, 7, 8, 16, 20, 5, 8, 9, 14, 6, 19, 14, 11, 9, 8, 9, 6, 10, 18, 18, 5]
value=[62, 93, 48, 23, 100, 75, 52, 22, 65, 42, 79, 83, 73, 85, 94, 31, 29, 65, 87, 52, 92, 48, 60, 67, 24, 98, 43, 68, 32, 7, 21, 99, 9, 86, 98, 60, 95, 98, 84, 54, 27, 66, 66, 97, 50, 37, 12, 66, 57, 96, 47, 5, 74, 80, 77, 12, 62, 87, 85, 96, 8, 53, 34, 22, 17, 89, 13, 64, 88, 77, 94, 39, 8, 66, 8, 29, 48, 18, 28, 94, 31, 89, 79, 31, 11, 73, 96, 79, 89, 24, 33, 74, 99, 8, 92, 36, 14, 78, 31, 26, 61, 63, 82, 17, 17, 20, 6, 25, 100, 95, 36, 18, 51, 75, 33, 81, 57, 23, 98, 83, 95, 6, 80, 45, 14, 53, 36, 32, 82, 22, 11, 25, 98, 53, 85, 40, 54, 21, 17, 39, 40, 86, 11, 41, 25, 15, 15, 84, 91, 35, 44, 23, 72, 61, 67, 22, 78, 30, 25, 97, 26, 42, 73, 96, 99, 38, 96, 58, 19, 23, 38, 47, 77, 21, 8, 90, 43, 26, 76, 56, 30, 31, 86, 29, 69, 8, 66, 54, 31, 67, 41, 57, 19, 42, 37, 17, 88, 7, 10, 44]


maxWeight = 2000

popSize = 200
count=1
for s in range(5):
  a=time.time()
  arr=knapsack(weight, value, maxWeight, popSize,10,200,0.1)
  b=time.time()


  '''print("Inputs:")
  print("Maximum weight of Sack:",maxWeight)
  print("Weight Array:",weight)
  print("Profit Array:",value)
  print("FINAL SOLUTION: " + str(arr))'''
  print("TOTAL PROFIT by GENETIC ALGORITHM: ",profit(arr,value))
  print("TOTAL TIME ELAPSED: ",b-a)