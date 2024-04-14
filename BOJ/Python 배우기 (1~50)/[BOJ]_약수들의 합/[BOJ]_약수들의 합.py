while True:
    약수=[]
    n = int(input())
    if n == -1:
        break
    for i in range(1,n):
        if n%i == 0:
            약수.append(i)
    if sum(약수) == n:
        문자=f'{n} = '
        for j in range(len(약수)):
            if j == 0:
                문자 += f'{약수[j]}'
            else:
                문자 += f' + {약수[j]}'
        print(문자)
    else:
        print(f'{n} is NOT perfect.')