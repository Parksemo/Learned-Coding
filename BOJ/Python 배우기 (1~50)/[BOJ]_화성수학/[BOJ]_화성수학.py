T = int(input())
for i in range(T):
    l = input().split()
    a = float(l[0])
    l.pop(0)
    for j in l:
        if j == '@':
            a = a * 3
        elif j == '%':
            a = a + 5
        elif j == '#':
            a = a - 7
    print(f"{a:.2f}")
    a=0
    l=[]