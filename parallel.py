import numpy as np
import multiprocessing as mp
from time import time

def func(arr):
    count=0
    for i in range(len(arr)):
        #brr.append(arr[i])
        for j in range(len(arr[0])):
            if arr[i][j]<5 and arr[i][j]>1:
                count+=1
    #print(brr)
    return count

arr=np.random.randint(0,10,size=[10000,5])
count=0
pool=mp.Pool(mp.cpu_count())
a=time()
brr=[]
for i in range(len(arr)):
    #brr.append(arr[i])
    for j in range(len(arr[0])):
        if arr[i][j]<5 and arr[i][j]>1:
            count+=1
b=time()
#print(arr)
c=time()
result=pool.apply(func,args=(arr))
#p=mp.Process(target=func,args=(arr))
#p.start()
#p.join()
d=time()
#pool.close()
print("Serial Execution:")
print(count)
print(b-a)
print("Parallel Execution:")
print(result)
print(d-c)