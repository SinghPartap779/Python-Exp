
def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0: return False
    return True

def factors(n):
    return [i for i in range(1, n+1) if n % i == 0]

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
