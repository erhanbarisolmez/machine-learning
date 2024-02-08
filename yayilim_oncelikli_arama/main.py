grafik = {
    'A': ['B','C'],
    'B': ['D','E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': [],
}

ziyaret = []
yigin=[]

def bfs(ziyaret,grafik,node):
    ziyaret.append(node)
    yigin.append(node)

    while yigin:
        s=yigin.pop(0)
        print(s,end=" ")

        for komsu in grafik[s]:
            if komsu not in ziyaret:
                yigin.append(komsu)

bfs(ziyaret, grafik, 'A')