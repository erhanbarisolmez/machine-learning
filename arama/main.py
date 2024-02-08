# Yayılım öncelikli arama
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

# Derin öncelikli arama
ziyaret=set()

def dfs(ziyaret,grafik,node):
    if node not in ziyaret:
        print(node)
        ziyaret.add(node)
        for komsu in grafik[node]:
            dfs(ziyaret, grafik, komsu)

dfs(ziyaret, grafik, 'A')

