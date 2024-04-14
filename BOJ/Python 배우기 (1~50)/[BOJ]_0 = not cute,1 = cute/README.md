## 0 = not cute / 1 = cute
[문제링크](https://www.acmicpc.net/problem/10886)

![0 = not cute / 1 = cute](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D0%20=%20not%20cute,%201%20=%20cute.JPG?raw=true)
<br>
```python
N = int(input())
not_cute = 0
cute = 0
for _ in range(N): # 설문조사를 한 사람의 수 N
    의견 = input() # 각 사람이 설문 조사에 표명한 의견
    if 의견 == '0': # 0은 준희가 귀엽지 않다고 했다는 뜻
        not_cute += 1
    elif 의견 == '1': # 1은 준희가 귀엽다고 했다는 뜻
        cute += 1    
if cute < not_cute: # 귀엽지 않다는 의견이 더 많을 경우 "Junhee is not cute!"를 출력
    print('Junhee is not cute!')
elif cute > not_cute: # 귀엽다는 의견이 많을 경우 "Junhee is cute!"를 출력
    print('Junhee is cute!')
```