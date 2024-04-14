N = int(input())
not_cute = 0
cute = 0
for _ in range(N):
    의견 = input()
    if 의견 == '0':
        not_cute += 1
    elif 의견 == '1':
        cute += 1    
if cute < not_cute:
    print('Junhee is not cute!')
elif cute > not_cute:
    print('Junhee is cute!')