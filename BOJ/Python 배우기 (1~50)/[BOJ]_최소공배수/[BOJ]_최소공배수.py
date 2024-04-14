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