from audioop import minmax
import ff
import networkx as nx

G = nx.Graph()

G.add_edge("s", "a", sim=1)
G.add_edge("s", "b", sim=5)
G.add_edge("a", "b", sim=6)
G.add_edge("a", "t", sim=2)
G.add_edge("b", "t", sim=3)

sp_list = ff.ford_fulkerson(G, "s", "t")

print(sp_list)