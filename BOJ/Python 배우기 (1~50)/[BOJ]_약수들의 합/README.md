## 약수들의 합
[문제링크](https://www.acmicpc.net/problem/9506)

![약수들의 합](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EC%95%BD%EC%88%98%EB%93%A4%EC%9D%98%20%ED%95%A9.JPG?raw=true)
<br>
```python
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
```