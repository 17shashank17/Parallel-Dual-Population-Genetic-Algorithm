import random
import time
import concurrent.futures

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
  fitness = getFitness(pop, weight, value, MAX)
  reserve_fitness=getFitness(reserve_pop,weight,value,MAX)
  while(not test(fitness, percent) and generation < maxGen):
    generation += 1
    temp=pop.copy()
    with concurrent.futures.ProcessPoolExecutor() as executer:
        f1=executer.submit(newPopulation,pop,fitness,mut)
        f2=executer.submit(newPopulation,reserve_pop,reserve_fitness,mut)
        pop=f1.result()
        reserve_pop=f2.result()
    
    #pop = newPopulation(pop, fitness, mut)
    #reserve_pop=newPopulation(reserve_pop,reserve_fitness,mut)
    with concurrent.futures.ProcessPoolExecutor() as executer1:
        f1=executer1.submit(getFitness,pop,weight,value,MAX)
        f2=executer1.submit(getFitness,reserve_pop,weight,value,MAX)
        fitness=f1.result()
        reserve_fitness=f2.result()
    #fitness = getFitness(pop, weight, value, MAX)
    #reserve_fitness=getFitness(reserve_pop,weight,value,MAX)
    pop=crossBreeding(pop,reserve_pop,fitness,reserve_fitness)
    reserve_pop=temp.copy()
    #with concurrent.futures.ProcessPoolExecutor() as executer2:
    #    f1=executer2.submit(getFitness,pop,weight,value,MAX)
    #    fitness=f1.result()
    fitness = getFitness(pop, weight, value, MAX)

  arr=selectElite(pop, fitness)
  return arr

def generate(value, popSize):
  length = len(value)
  pop = [[random.randint(0,1) for i in range(length)] for j in range(popSize)]
  return pop
  
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

#weight=[20, 15, 16, 13, 11, 11, 16, 16, 10, 7, 11, 9, 17, 15, 15, 11, 13, 11, 14, 13, 20, 18, 6, 9, 8, 5, 14, 18, 19, 15, 5, 5, 12, 6, 17, 12, 20, 14, 17, 9, 17, 5, 11, 17, 14, 18, 15, 16, 8, 20, 13, 19, 15, 15, 18, 15, 12, 15, 11, 17, 13, 19, 15, 17, 10, 13, 10, 7, 14, 14, 12, 5, 8, 18, 12, 8, 10, 7, 9, 18, 7, 13, 15, 6, 13, 7, 11, 16, 6, 6, 10, 8, 18, 7, 13, 12, 9, 19, 18, 20, 19, 12, 7, 15, 12, 16, 20, 17, 12, 19, 7, 20, 6, 9, 10, 20, 16, 9, 10, 19, 6, 16, 13, 7, 8, 17, 10, 11, 7, 18, 5, 14, 16, 18, 15, 8, 11, 11, 18, 7, 6, 6, 18, 14, 18, 5, 10, 20, 20, 5, 14, 9, 19, 17, 17, 9, 16, 18, 11, 20, 15, 6, 9, 15, 18, 16, 8, 6, 19, 8, 10, 18, 20, 7, 18, 20, 15, 5, 17, 9, 7, 8, 16, 20, 5, 8, 9, 14, 6, 19, 14, 11, 9, 8, 9, 6, 10, 18, 18, 5]
#value=[62, 93, 48, 23, 100, 75, 52, 22, 65, 42, 79, 83, 73, 85, 94, 31, 29, 65, 87, 52, 92, 48, 60, 67, 24, 98, 43, 68, 32, 7, 21, 99, 9, 86, 98, 60, 95, 98, 84, 54, 27, 66, 66, 97, 50, 37, 12, 66, 57, 96, 47, 5, 74, 80, 77, 12, 62, 87, 85, 96, 8, 53, 34, 22, 17, 89, 13, 64, 88, 77, 94, 39, 8, 66, 8, 29, 48, 18, 28, 94, 31, 89, 79, 31, 11, 73, 96, 79, 89, 24, 33, 74, 99, 8, 92, 36, 14, 78, 31, 26, 61, 63, 82, 17, 17, 20, 6, 25, 100, 95, 36, 18, 51, 75, 33, 81, 57, 23, 98, 83, 95, 6, 80, 45, 14, 53, 36, 32, 82, 22, 11, 25, 98, 53, 85, 40, 54, 21, 17, 39, 40, 86, 11, 41, 25, 15, 15, 84, 91, 35, 44, 23, 72, 61, 67, 22, 78, 30, 25, 97, 26, 42, 73, 96, 99, 38, 96, 58, 19, 23, 38, 47, 77, 21, 8, 90, 43, 26, 76, 56, 30, 31, 86, 29, 69, 8, 66, 54, 31, 67, 41, 57, 19, 42, 37, 17, 88, 7, 10, 44]

weight=[19, 7, 18, 9, 8, 13, 11, 15, 14, 20, 20, 12, 10, 6, 7, 11, 9, 20, 16, 9, 12, 19, 10, 19, 14, 19, 16, 13, 12, 13, 13, 5, 8, 18, 9, 14, 11, 14, 9, 17, 15, 18, 15, 20, 10, 20, 15, 14, 18, 15, 10, 18, 5, 10, 6, 18, 13, 6, 11, 15, 7, 5, 13, 13, 8, 6, 14, 17, 5, 9, 16, 16, 12, 5, 20, 16, 16, 6, 7, 20, 6, 20, 14, 15, 5, 20, 7, 13, 15, 17, 20, 6, 6, 7, 7, 20, 19, 18, 11, 5, 13, 13, 18, 14, 9, 6, 13, 6, 7, 16, 15, 11, 14, 15, 12, 20, 20, 7, 11, 10, 11, 12, 7, 20, 8, 7, 16, 7, 18, 16, 14, 5, 10, 11, 16, 16, 9, 7, 11, 15, 5, 10, 7, 6, 17, 19, 16, 5, 14, 13, 5, 13, 8, 17, 15, 16, 15, 17, 12, 14, 12, 8, 18, 18, 6, 18, 6, 5, 20, 18, 14, 17, 19, 18, 10, 18, 14, 5, 13, 6, 16, 14, 15, 7, 12, 8, 8, 18, 9, 19, 20, 7, 14, 11, 12, 18, 18, 20, 10, 10, 8, 10, 8, 13, 16, 19, 8, 17, 5, 16, 11, 20, 6, 7, 11, 5, 19, 19, 12, 6, 8, 19, 7, 17, 15, 8, 15, 13, 8, 8, 7, 16, 6, 20, 12, 6, 11, 6, 7, 10, 9, 14, 14, 19, 9, 16, 9, 10, 19, 6, 16, 6, 12, 6, 7, 11, 5, 17, 19, 19, 15, 18, 11, 7, 19, 6, 10, 11, 17, 6, 5, 19, 7, 5, 15, 17, 7, 20, 13, 6, 7, 6, 17, 20, 9, 8, 19, 16, 12, 8, 8, 8, 9, 14, 18, 14, 11, 10, 5, 20, 12, 18, 9, 7, 10, 14, 7, 20, 19, 7, 9, 5, 20, 16, 13, 7, 9, 9, 15, 6, 5, 14, 17, 17, 6, 10, 18, 20, 9, 19, 20, 12, 20, 8, 14, 9, 7, 16, 8, 12, 9, 20, 6, 14, 5, 13, 10, 10, 7, 16, 19, 9, 18, 12, 6, 12, 14, 16, 13, 11, 14, 17, 6, 11, 7, 10, 19, 6, 9, 11, 18, 5, 9, 11, 10, 8, 12, 18, 12, 8, 15, 13, 19, 19, 15, 20, 5, 15, 20, 8, 19, 17, 13, 15, 17, 13, 13, 10, 9, 16, 11, 14, 8, 12, 11, 19, 17, 16, 5, 5, 12, 16, 20, 5, 14, 15, 7, 8, 13, 19, 18, 16, 10, 12, 11, 8, 11, 17, 12, 17, 7, 5, 12, 19, 10, 6, 12, 8, 16, 13, 10, 5, 18, 16, 11, 12, 13, 13, 17, 14, 10, 9, 11, 11, 6, 5, 11, 19, 15, 10, 7, 16, 20, 7, 14, 18, 7, 5, 16, 12, 9, 10, 8, 5, 7, 7, 16, 9, 19, 7, 13, 20, 19, 11, 6, 10, 13, 13, 13, 19, 14, 14, 14, 11, 10, 6, 8, 10, 17, 13]
value=[29, 49, 49, 91, 40, 37, 94, 36, 15, 98, 44, 99, 95, 70, 32, 46, 20, 55, 65, 90, 25, 21, 97, 9, 18, 28, 44, 7, 85, 15, 96, 69, 61, 86, 95, 19, 97, 18, 79, 45, 88, 73, 91, 77, 98, 77, 59, 91, 41, 34, 98, 63, 94, 15, 9, 89, 32, 66, 28, 21, 9, 31, 87, 98, 57, 91, 16, 48, 42, 56, 42, 87, 10, 8, 79, 32, 52, 100, 18, 81, 42, 57, 95, 86, 94, 86, 69, 42, 65, 20, 24, 35, 92, 65, 57, 23, 55, 23, 43, 6, 13, 17, 73, 29, 47, 83, 50, 47, 98, 96, 15, 7, 28, 28, 65, 13, 12, 16, 60, 47, 85, 34, 81, 83, 50, 10, 41, 88, 29, 6, 24, 49, 9, 85, 24, 10, 37, 72, 15, 24, 26, 96, 9, 39, 5, 36, 25, 59, 54, 83, 99, 43, 52, 63, 16, 38, 13, 63, 30, 61, 15, 58, 51, 97, 10, 45, 44, 30, 74, 66, 93, 32, 31, 37, 9, 43, 63, 15, 90, 67, 18, 12, 58, 100, 38, 8, 89, 72, 81, 7, 92, 28, 85, 71, 62, 8, 19, 26, 32, 57, 30, 65, 78, 62, 17, 28, 49, 48, 91, 25, 83, 54, 70, 58, 21, 47, 12, 16, 82, 15, 98, 11, 93, 75, 48, 64, 100, 7, 32, 27, 54, 11, 27, 57, 87, 30, 40, 67, 89, 75, 92, 94, 14, 22, 23, 8, 65, 37, 74, 16, 89, 14, 76, 74, 89, 59, 43, 30, 90, 22, 76, 96, 98, 39, 24, 18, 26, 87, 31, 77, 87, 63, 81, 81, 30, 54, 33, 49, 22, 20, 27, 12, 46, 83, 72, 8, 29, 77, 22, 26, 55, 19, 79, 11, 99, 67, 19, 84, 86, 78, 36, 30, 31, 44, 40, 13, 47, 11, 42, 23, 94, 82, 85, 31, 24, 63, 84, 18, 34, 31, 9, 67, 44, 82, 82, 55, 46, 65, 33, 83, 23, 68, 51, 20, 27, 60, 77, 68, 23, 24, 88, 97, 92, 67, 52, 12, 9, 94, 68, 25, 64, 25, 59, 47, 39, 23, 9, 80, 10, 64, 38, 45, 5, 68, 13, 16, 95, 78, 9, 77, 74, 23, 37, 26, 37, 91, 65, 78, 99, 80, 99, 87, 14, 13, 44, 75, 60, 37, 32, 83, 49, 55, 55, 51, 40, 73, 74, 76, 5, 42, 19, 13, 41, 48, 8, 46, 76, 53, 23, 93, 17, 35, 28, 88, 40, 99, 76, 39, 34, 49, 77, 60, 59, 63, 58, 52, 67, 92, 20, 28, 85, 92, 88, 99, 68, 59, 34, 79, 78, 29, 31, 44, 38, 46, 22, 28, 92, 28, 73, 19, 90, 51, 86, 10, 10, 25, 78, 76, 91, 99, 78, 25, 58, 70, 6, 53, 21, 71, 46, 47, 27, 71, 42, 64, 72, 38, 22, 80, 45, 68, 35, 95, 48, 11, 68, 79, 97, 10, 92, 83, 80, 8, 75, 12, 93, 42, 80, 6, 10, 26]


maxWeight = 5000

popSize = 300
count=1
for s in range(5):
  a=time.time()
  arr=knapsack(weight, value, maxWeight, popSize,10,300,0.1)
  b=time.time()


  '''print("Inputs:")
  print("Maximum weight of Sack:",maxWeight)
  print("Weight Array:",weight)
  print("Profit Array:",value)
  print("FINAL SOLUTION: " + str(arr))'''
  print("TOTAL PROFIT by GENETIC ALGORITHM: ",profit(arr,value))
  print("TOTAL TIME ELAPSED: ",b-a)