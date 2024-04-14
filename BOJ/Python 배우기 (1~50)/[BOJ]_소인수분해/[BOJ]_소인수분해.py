N = int(input())
l=[]
n=2
while True:
    if N == 1:
        break
    elif N%n == 0:
        N = N//n
        l.append(n)
    else:
        n +=1
for i in l:
    print(i)