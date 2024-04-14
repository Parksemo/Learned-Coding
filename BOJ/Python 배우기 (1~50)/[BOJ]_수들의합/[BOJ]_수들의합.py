S = int(input())
N=1
while True:
    S-=N
    if S<=N:
        break
    N+=1
print(N)