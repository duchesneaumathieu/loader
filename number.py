import numpy as np

def powermod(k,i,n):
    if i==0: return 1
    if i%2==0: return powermod(k, i/2, n)**2 % n
    if i%2==1: return (k*powermod(k, i-1, n)) % n

def factorization(n):
    d = dict()
    for i in range(2, int(np.sqrt(n))+1):
        if n % i == 0: d[i] = 0
        while n % i == 0:
            n = n / i
            d[i] += 1
    if n > 1:
        if n in d: d[n] += 1
        else: d[n] = 1
    return d

def bmg(p, start=None, skip=0):
    """
    Biggest Modular Generator
    """
    count_gen=0
    a = p - 2 if start is None else min(p - 2, start)
    while a > 1:
        if isprimroot(p, a):
            if count_gen == skip:
                return a
            count_gen += 1
        a -= 1
    return None

def isprimroot(p, m):
    for f in factorization(p-1).keys():
        if powermod(m, (p-1)/f, p) == 1:
            return False
    return True

def RabinMiller(n, k):
    """
    Input1: n, an integer to be tested for primality;
    Input2: k, a parameter that determines the accuracy of the test
    Output: False if n is composite, otherwise True if probably prime
    """
    if n<2: return False
    if n in [2,3]: return True
    if n%2==0: return False
    
    m = n-1
    r=0
    while(m%2==0):
        r+=1
        m = m/2
    d = (n-1)/2**r
    
    for _ in range(k): #WitnessLoop
        a = np.random.randint(2, n-2)
        x = powermod(a, d, n)
        if x == 1 or x == n - 1: continue
        
        continue_WitnessLoop = False
        for _ in range(r-1):
            x = (x**2) % n
            if x == 1:
                return False
            if x == n - 1:
                continue_WitnessLoop = True
                continue
        if continue_WitnessLoop: continue
        return False
    return True

def next_prime(k=16, start=0, skip=0):
    if start in [0,1]: m = 2
    elif start%2==0: m = start+1
    else: m = start
    while not RabinMiller(m, k): m += 2
    return m if skip == 0 else next_prime(k=k, start=m, skip=skip-1)