import heapq
import datetime


class Dijkstra:
    def __init__ (self, env, start_node, end_node):
        self.env = env
        self.env.set_start_end(start_node, end_node)  # call set_start_end() in env to set the start and end node


    def reset(self):
        self.cost = {node: float('inf') for node in self.env.nodes}  # set to infinity
        self.predecessor = {node: None for node in self.env.nodes}
        self.cost[self.env.start_node] = 0  # cost from start node to itself is 0
        self.priority_queue = [(0, self.env.start_node)]


    def calculate_cost(self, current_cost, adj_edge):
        if self.env.evaluation in ("time"):
            cost = current_cost + self.env.get_edge_time(adj_edge)
        else:  # in ("distance")
            cost = current_cost + self.env.get_edge_distance(adj_edge)
        return cost


    # main function in dijkstra
    def search(self):
        print('Search Start...')
        start_time = datetime.datetime.now()

        self.reset()  # the initial state of the algorithm

        while self.priority_queue:
            current_cost, current_node = heapq.heappop(self.priority_queue) # get the minimum one from the heap

            # If the node is the end node, then stop searching
            if current_node == self.env.end_node:
                break

            # Explore the neighbors nodes
            for adj_edge in self.env.decode_node_to_edges(current_node, direction = 'outgoing'):
                # Corresponding neighbor node
                adj_node = self.env.decode_edge_to_node(adj_edge, direction = 'end')
                # Calculate the cost of the neighbor
                temp_cost = self.calculate_cost(current_cost, adj_edge)

                # If the tentative distance is less than the current distance of the neighbor
                if temp_cost < self.cost[adj_node]:
                    self.cost[adj_node] = temp_cost # update the distance of the neighbor
                    self.predecessor[adj_node] = current_node # update the predecessor of the neighbor
                    heapq.heappush(self.priority_queue, (temp_cost, adj_node)) # add the neighbor to the priority queue

        # Construct the path from the start node to the goal node
        node_path = []
        edge_path = []

        temp_node = self.env.end_node
        while temp_node is not None:
            node_path.append(temp_node)
            temp_node = self.predecessor[temp_node]
        node_path.reverse()

        for index in range(len(node_path)-1):
            intersect_edge = set(self.env.decode_node_to_edges(node_path[index], "outgoing")) & set(self.env.decode_node_to_edges(node_path[index+1], "incoming")) # find the edge between two nodes by knowing their set
            edge_path.append(next(iter(intersect_edge)))  # temp_edge is a set of elements (only one element here), so use it to get the element rather than the set by change it into iterator and get the next one because of its unordered prop

        # time the search process
        end_time = datetime.datetime.now()
        time_difference = end_time - start_time
        processing_seconds = time_difference.total_seconds()  # get second rather than timedelta object

        # --- results output ---
        print('Search Completed...\n')
        print(f'-- States: {node_path}\n')
        print(f'-- Edges: {edge_path}\n')  # states is nodes actually
        print(f'-- Processing Time: {processing_seconds} seconds')

        if self.env.evaluation in ("time"):
            print(f'-- Travelled Time: {round(( self.env.get_edge_time(edge_path) + self.env.get_tl_offset(edge_path) )/60, 2)} mins')
        else:  # in ("distance")
            print(f'-- Travelled Distance: {round(self.env.get_edge_distance(edge_path), 2)} m')

        return node_path, edge_path
