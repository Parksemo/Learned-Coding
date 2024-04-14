## 최소공배수
[문제링크](https://www.acmicpc.net/problem/1934)

![최소공배수](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EC%B5%9C%EC%86%8C%EA%B3%B5%EB%B0%B0%EC%88%98.JPG?raw=true)
<br>
<처음 풀이했던 코드>
```python
T = int(input())
for i in range(T):
    A, B = map(int,input().split())
    공배수 = []
    for j in range(1,B+1):
        for k in range(1,A+1):
            if k*B == j*A:
                공배수.append(k*B)
    print(공배수[0])
```
위 코드 처럼 두 자연수 A와 B를 받은 후 for 중첩문을 통해 A와 B간의 배수가 같을 때 공배수를 리스트에 추가하였고, 거기서 처음으로 추가된 공배수가 최소공배수일 것이라고 판단하여 코드를 작성하였는데, 제출하니 시간초과가 되었습니다.
<br>
또 다른 방법으로 검색하여 참조해본 결과, 다음과 같은 코드를 작성하였습니다.
```python
def 최대공약수(A, B):
  if B == 0:
    return A
  else:
    return 최대공약수(B, A%B)
  
def 최소공배수(A, B):
  result = (A*B) // 최대공약수(A,B)
  return result

T = int(input())

for i in range(T):
  A, B = map(int, input().split())
  print(최소공배수(A, B))
```
이는 최소 공배수는 두 수를 곱한 값에 최대공약수를 나눈 값과 동일하며, 최대공약수는 유클리드 호제법을 이용하여 구할 수 있다.

![최소공배수](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EC%B5%9C%EC%86%8C%EA%B3%B5%EB%B0%B0%EC%88%98(%EC%9C%A0%ED%81%B4%EB%A6%AC%EB%93%9C%ED%98%B8%EC%A0%9C%EB%B2%95).JPG?raw=true)
[출처-나무위키](https://ko.wikipedia.org/wiki/%EC%9C%A0%ED%81%B4%EB%A6%AC%EB%93%9C_%ED%98%B8%EC%A0%9C%EB%B2%95)