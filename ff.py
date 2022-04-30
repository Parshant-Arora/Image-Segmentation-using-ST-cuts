import numpy as np

def bfs(graph, s, t, parent, row):
        visited = [False] * (row)
        queue = []

        queue.append(s)
        visited[s] = True

        while queue:
            u = queue.pop(0)
            for ind, val in enumerate(graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        return True if visited[t] else False
    
def dfs(graph, V, s, visited):
    stack = [s]
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            stack.extend([u for u in range(V) if graph[v][u]])

    
    

def ford_fulkerson(G, source, sink):
        parent = [-1] * (len(list(G.nodes)))
        max_flow = 0
        node_to_label = {source : 0}
        label_to_node = {0 : source}
        for ind , nodes in enumerate(list(G.nodes)):
            if nodes not in node_to_label :
                node_to_label[nodes] = ind
                label_to_node[ind] = nodes
        
        graph = np.zeros([len(list(G.nodes)),len(list(G.nodes))])

        for u,v in G.edges:
            graph[node_to_label[u]][node_to_label[v]] = G[u][v]['sim']
            graph[node_to_label[v]][node_to_label[u]] = G[u][v]['sim']

        while bfs(graph, node_to_label[source], node_to_label[sink], parent, len(list(G.nodes))):

            path_flow = float("Inf")
            s = node_to_label[sink]
            while(s != node_to_label[source]):
                path_flow = min(path_flow, graph[parent[s]][s])
                s = parent[s]
                
                

            max_flow += path_flow
            v = node_to_label[sink]
            while(v != node_to_label[source]):
                u = parent[v]
                graph[u][v] -= path_flow
                graph[v][u] += path_flow
                v = parent[v]

        V = len(list(G.nodes))
        visited = np.zeros(V, dtype=bool)
        SP_list = []
        dfs(graph, V, s, visited)
        for i in range(V):
            if visited[i]:
                SP_list.append(label_to_node[i])
        return SP_list