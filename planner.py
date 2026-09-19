import heapq
import time
from typing import List, Tuple, Dict, Set
from models import Node
from environment import SurgicalLabEnvironment

class ForkliftPlanner:
    """A* path planner comparing different heuristics"""
    def __init__(self, environment: SurgicalLabEnvironment):
        self.env = environment
        
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int], heuristic_type: str = 'manhattan') -> Tuple[List[Tuple[int, int]], int, float]:
        start_time = time.time()
        nodes_expanded = 0
        
        # ---------------------------------------------------------
        # SETUP: We have initialized the starting node for you!
        # ---------------------------------------------------------
        start_node = Node(start[0], start[1])
        start_node.g = 0
        
        # Choose the correct heuristic based on the function argument
        if heuristic_type == 'manhattan':
            start_node.h = self.env.manhattan_distance(start[0], start[1], goal[0], goal[1])
        else:
            start_node.h = self.env.euclidean_distance(start[0], start[1], goal[0], goal[1])
            
        start_node.f = start_node.g + start_node.h
        
        # Data Structures you will need
        open_set = []
        heapq.heappush(open_set, start_node)
        
        closed_set: Set[Node] = set()
        open_dict: Dict[Tuple[int, int], Node] = {(start[0], start[1]): start_node}
        
        # ---------------------------------------------------------
        # TODO 1: THE CORE LOOP
        # Loop as long as there are nodes in the open_set
        # ---------------------------------------------------------
        
        while open_set:
            # 1. Pop the node with the lowest f-score from open_set using heapq
            current = heapq.heappop(open_set)

            # 2. (Optional but recommended) Check if this node is stale using open_dict
            latest = open_dict.get((current.x, current.y))
            if latest is not None and latest is not current:
                continue
            if current in closed_set:
                continue

            # 3. Check if the current node is the goal! 
            if (current.x, current.y) == goal:
                path = []
                node = current
                while node:
                    path.append((node.x, node.y))
                    node = node.parent
                path.reverse()
                return path, nodes_expanded, time.time() - start_time
            
            # If yes, reconstruct the path using the .parent attributes, reverse it, and return it.
            
            # 4. Add the current node to the closed_set and increment nodes_expanded
            closed_set.add(current)
            nodes_expanded += 1

            # ---------------------------------------------------------
            # TODO 2: EVALUATING NEIGHBORS
            # ---------------------------------------------------------
            # 5. Use self.env.get_neighbors(current.x, current.y) to get valid moves
            
            # For each neighbor:
            for nx, ny in self.env.get_neighbors(current.x, current.y):
                neighbor_key = (nx, ny)

                # a. Skip if it is already in the closed_set
                if Node(nx, ny) in closed_set:
                    continue

                # b. Calculate the tentative_g score (current g + 1)
                tentative_g = current.g + 1

                existing = open_dict.get(neighbor_key)

                # c. If it's a new node OR the tentative_g is better than its existing g:
                if existing is None or tentative_g < existing.g:
                    neighbor_node = Node(nx, ny)
                    neighbor_node.parent = current
                    neighbor_node.g = tentative_g

                    if heuristic_type == 'manhattan':
                        neighbor_node.h = self.env.manhattan_distance(nx, ny, goal[0], goal[1])
                    elif heuristic_type == 'chebyshev':
                        neighbor_node.h = self.env.chebyshev_distance(nx, ny, goal[0], goal[1])
                    else:
                        neighbor_node.h = self.env.euclidean_distance(nx, ny, goal[0], goal[1])

                    neighbor_node.f = neighbor_node.g + neighbor_node.h

                    heapq.heappush(open_set, neighbor_node)
                    open_dict[neighbor_key] = neighbor_node              
	
        # Returns empty list if no path is found
        return [], nodes_expanded, time.time() - start_time
