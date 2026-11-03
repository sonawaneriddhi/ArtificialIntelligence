from collections import deque
import time

graph = {
'A': ['B', 'C'],
'B': ['A', 'D', 'E'],
'C': ['A', 'F'],
'D': ['B'],
'E': ['B', 'F'],
'F': ['C', 'E', 'G'],
'G': ['F']
}

start = 'A'

goal = 'G'

def reconstruct_path(parent, goal):
    if goal not in parent:
        return []
    
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path

def bfs(graph, start, goal):
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    nodes_explored = 0

    while queue:
        current = queue.popleft()
        nodes_explored += 1

        if current == goal:
            path = reconstruct_path(parent, goal)
            return path, nodes_explored
        
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    return [], nodes_explored

bfs_path, bfs_nodes = bfs(graph, start, goal)
print("BFS Path:", bfs_path)
print("BFS Path Length:", len(bfs_path) - 1)
print("BFS Nodes Explored:", bfs_nodes)

def dfs(graph, start, goal):
    stack = [start]
    visited = {start}
    parent = {start: None}
    nodes_explored = 0

    while stack:
        current = stack.pop()
        nodes_explored += 1

        if current == goal:
            path = reconstruct_path(parent, goal)
            return path, nodes_explored
        
        for neighbor in reversed(graph[current]):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                stack.append(neighbor)

    return [], nodes_explored

dfs_path, dfs_nodes = dfs(graph, start, goal)
print("DFS Path:", dfs_path)
print("DFS Path Length:", len(dfs_path) - 1)
print("DFS Nodes Explored:", dfs_nodes)

# --------- BFS Timing -----------

start_time = time.perf_counter()
bfs_path, bfs_nodes = bfs(graph, start, goal)
bfs_time = time.perf_counter() - start_time
bfs_length = len(bfs_path) - 1
print("BFS Path:", bfs_path)
print("BFS Path Length:", bfs_length)
print("BFS Nodes Explored:", bfs_nodes)
print("BFS Execution Time:", bfs_time, "seconds")

# ---------- DFS Timing ------------

start_time = time.perf_counter()
dfs_path, dfs_nodes = dfs(graph, start, goal)
dfs_time = time.perf_counter() - start_time
dfs_length = len(dfs_path) - 1
print("DFS Path:", dfs_path)
print("DFS Path Length:", dfs_length)
print("DFS Nodes Explored:", dfs_nodes)
print("DFS Execution Time:", dfs_time, "seconds")
