import sys
N, K = map(int,sys.stdin.readline().split())
N_l = list(range(1,N+1))

if K == 1:
    print('<',end='')
    print(*N_l, sep=', ', end='')
    print('>')

elif K != 1:
    K = K-1
    KK = K

    N_O = []
    while True:
        if len(N_l) > KK:
            N_O.append(N_l.pop(KK))
        elif len(N_l) <= KK:
            KK = KK%len(N_l)
            N_O.append(N_l.pop(KK))
        KK = KK + K

        if len(N_l) == 1:
            N_O.append(N_l.pop())
            break
    print('<',end='')
    print(*N_O, sep=', ', end='')
    print('>')