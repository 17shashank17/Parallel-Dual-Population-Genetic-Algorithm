import numpy as np
from time import time
import pymp as py
from random import randint

arr=[randint(1,10) for i in range(1000000)]
brr=arr.copy()
sum=0
sum_s=0
c=time()
for i in range(len(arr)):
    arr[i]=arr[i]**2
d=time()
print(d-c)
a=time()
with py.Parallel(2) as p:
    for i in range(1000000):
        brr[i]=brr[i]*brr[i]
b=time()

print(b-a)
