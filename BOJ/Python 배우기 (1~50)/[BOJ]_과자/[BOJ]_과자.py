K, N, M = map(int,input().split())
부모님께_받아야_하는_돈 = K*N-M
if 부모님께_받아야_하는_돈 >= 0:
  print(K*N-M)
else:
  print(0)