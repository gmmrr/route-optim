import sys
import sumolib
import math
import networkx as nx
import matplotlib.pyplot as plt
import random


class traffic_env:
    def __init__ (self, network_file, congestion = [], traffic_light = [], evaluation = "", congestion_level = "", travel_speed=80):
        # 1. Define network_file
        self.network_file = network_file  # read the file
        self.net = sumolib.net.readNet(network_file)  # file -> net
        self.nodes = [node.getID().upper() for node in self.net.getNodes()]  # net -> nodes
        self.edges = [edge.getID() for edge in self.net.getEdges()]  # net -> edges
        self.action_space = [0, 1, 2, 3]  #  0 Right, 1 Up, 2 Left, 3 Down, so note that each junction has less than 4 exit in this case
        self.state_space = self.nodes  # state_space
        self.edge_label = self.decode_edges_to_label()  # give every edges a label by their direction in the aspect of x-y coordinate

        # 2. Define congestions edges with its original pattern [("gneF_I", 10), ...]
        if congestion:  # if congestion is defined
            self.congested_edges = [item[0] for item in congestion]  # "gneF_I" is an edge, where "gne" stands for "generic edge". It means from F to I for example
            self.congestion_duration = [item[1] for item in congestion]  # 10 is the duration of so called "traffic jam"

            for edge in self.congested_edges:  # make sure that all congested_edges are in the net
                if edge not in self.edges:
                    sys.exit(f'Error: Invalid congestion_edges {edge}')
            print(f'Traffic Congestion: {list(zip(self.congested_edges, self.congestion_duration))}')
            print(f'Num of Congested/All Edges: {len(self.congested_edges)}/{len(self.edges)}')

        else:  # if congestion is not defined, then randomly choose edges and its duration
            if congestion_level == "low":
                traffic_level = 0.05
            elif congestion_level == "medium":
                traffic_level = 0.10
            elif congestion_level == "high":
                traffic_level = 0.20
            self.congested_edges = random.sample(self.edges, round(len(self.edges) * traffic_level))
            self.congestion_duration = [random.randint(10, 20) for _ in range(len(self.congested_edges))]
            print(f'Traffic Congestion: {list(zip(self.congested_edges, self.congestion_duration))}')
            print(f'Num of Congested/All Edges: {len(self.congested_edges)}/{len(self.edges)}')

        # 3. Define traffic lights nodes, with the original pattern [ (["B", "C"], 5), ...]
        self.tl_nodes = [item[0] for item in traffic_light]  # ["B", "C"] is a list of nodes. if you meet a traffic light at B, you won't meet another one at C again
        self.tl_duration = [item[1] for item in traffic_light] # 5 is the duration

        for nodes_lst in self.tl_nodes:  # just to make sure if tl_nodes are defined in the net
            if isinstance(nodes_lst, str): # if it is "B" rather than ["B", "C"], than make ["B"]
                nodes_lst = [nodes_lst]
            for node in nodes_lst:
                if node not in self.nodes:
                    sys.exit(f'Error: Invalid traffic_lights node {node}')
        print(f'Traffic Light: {list(zip(self.tl_nodes, self.tl_duration))}')

        # 4. Define evaluation type
        if evaluation not in ('distance', 'time'):
            sys.exit('Error: Invalid evaluation type, provide only "distance" or "time"')
        self.evaluation = evaluation

        # 5. Define travel speed
        self.travel_speed = travel_speed


    # Set starting and ending nodes
    def set_start_end(self, start_node, end_node):
        """
        Called by dijkstra.py and agent.py to set the starting and ending nodes of the environment.

        Returns:
        - void
        """

        # Check if the nodes are valid
        if start_node not in self.nodes:
            sys.exit('Error: Invalid start node')
        elif end_node not in self.nodes:
            sys.exit('Error: Invalid end node')
        else:
            self.start_node = start_node
            self.end_node = end_node


    # Match node to edges
    def decode_node_to_edges(self, node, direction = None):
        """
        Given a node and direction, returns a list of edges associated with that node.

        Args:  #
        - node (str): The ID of the node to match to edges
        - direction (str or None): The direction of the edges to return.
            If None, all edges are returned. Otherwise, must be one of the following strings:
            - 'incoming': return only edges where the node is the end
            - 'outgoing': return only edges where the node is the start

        Returns:
        - A list of edges (str) associated with the given node, in the specified direction if specified.
        """

        # Check if the direction is valid
        if direction not in ('incoming', 'outgoing', None):
            sys.exit(f'Invalid direction: {direction}')

        edges = []
        net_node = self.net.getNode(node)

        # Match node and direction to return edges
        if direction == 'incoming':
            for edge in net_node.getIncoming():
                if edge.getToNode().getID() == node:
                    edges.append(edge.getID())

        elif direction == 'outgoing':
            for edge in net_node.getOutgoing():
                if edge.getFromNode().getID() == node:
                    edges.append(edge.getID())

        else:
            for edge in net_node.getIncoming() + net_node.getOutgoing():
                if edge.getToNode().getID() == node or edge.getFromNode().getID() == node:
                    edges.append(edge.getID())

        return edges


    # Label edges based of junction from ( 0 Right -> 1 Up -> 2 Left -> 3 Down )
    def decode_edges_to_label(self):
        """
        Iterates through the whole state space and returns a dictionary of each state and the direction it is headed.

        Returns:
        - A dictionary of states (str) matched with its direction.
        """

        edge_labelled = {edge: None for edge in self.edges}

        def get_edge_label(node, outgoing_edges):
            # get the nodes outgoing
            start_x, start_y = self.net.getNode(node).getCoord()

            # store edge angle
            edge_angle = []

            # through each outgoing edge
            for edge in outgoing_edges:
                # get the end node of the edge
                end_node = self.decode_edge_to_node(edge)
                end_x, end_y = self.net.getNode(end_node).getCoord()

                # get the delta_x and delta_y
                delta_x = end_x - start_x
                delta_y = end_y - start_y

                # get the angle by delta_x and delta_y
                angle = math.degrees(math.atan2(delta_y, delta_x))

                # save the edge and its corresponding angle
                edge_angle.append((edge, angle))

            # sort from 0 to 180 to -180 to 0 (Right -> Up -> Left -> Down -> Right)
            edge_angle = sorted(edge_angle, key=lambda x: ((x[1] >= 0) * -180, x[1]))

            # label edges
            for i in range(len(edge_angle)):
                edge_labelled[edge_angle[i][0]] = i  # edge_angle[i][0] is the ordered edge, btw [1] is its angle

        for node in self.nodes:
            outgoing_edges = self.decode_node_to_edges(node, 'outgoing')
            if outgoing_edges:
                get_edge_label(node, outgoing_edges)

        return edge_labelled  # note that two edges in opposite directions have different ID (viewed as different edges)


    # Find the actions from a given edges
    def decode_edges_to_actions(self, edges):
        """
        Translate a list of given edges to their actions

        Args:
        - edges (list): The list of edges to be translated

        Returns:
        - A list of actions (int)
        """

        # Check if edges is in the edges list
        for edge in edges:
            if edge not in self.edges:
                sys.exit(f'Error: Edge {edge} not in Edges Space')

        # Get the label of each edge
        edge_label = self.edge_label

        # Returns a list of actions
        actions_lst = []
        for action in self.action_space:
            if action in [edge_label[edge] for edge in edges]:
                actions_lst.append(action)
        return actions_lst


    # Find the correspoding edge by given an edge set from a node and action
    def decode_edges_action_to_edge(self, edges, action):
        """
        Compute the new edge from a given edges and action taken.

        Args:
        - edges (list): The list of edges to be translated, they should be from a same node
        - action (int): The action taken

        Returns:
        - The new edge (str) or None if no match is found.
        """

        # Check if edges is in the edges list
        for edge in edges:
            if edge not in self.edges:
                sys.exit(f'Error: Edge {edge} not in Edges Space')

        # Get the direction of each edge
        edge_label = self.edge_label

        for edge in edges:
            if edge_label[edge] == action:
                return edge
        return None


    # Find the end node from a given edge
    def decode_edge_to_node(self, search_edge, direction = 'end'):
        """
        Given an edge return the start or ending node of that edge

        Args:
        - search_edge (str): The edge to be computed
        - direction (str): The direction of the node to return
          - 'start': node is the start of the edge
          - 'end': node is the end of the edge (default)

        Returns:
        - The node (str)
        """

        # Check if edges is in the edges list
        if search_edge not in self.edges:
            sys.exit('Error: Edge not in Edges Space!')

        edge = self.net.getEdge(search_edge)

        if direction == 'start':
            node = edge.getFromNode().getID()

        elif direction == 'end':
            node = edge.getToNode().getID()

        return node


    # Find the total distance travelled by giving a single edge / an edge_path
    def get_edge_distance(self, travel_edges):
        """
        Calculates the cost (distance travelled) through the selected path

        Args:
        - travel_edges: The list of edges of the selected path

        Return:
        - total_distance (float): The total distance travelled
        """

        total_distance = 0

        if isinstance(travel_edges, str):  # make "" into [""]
            travel_edges = [travel_edges]

        for edge in travel_edges:
            # Check if edges are in the edges list
            if edge not in self.edges:
                sys.exit(f'Error: Edge {edge} not in Edges Space!')
            # Sum up the distance of each edges
            total_distance += self.net.getEdge(edge).getLength()

        return total_distance


    # Find the total time taken by giving a single edge / an edge_path
    def get_edge_time(self, travel_edges):
        """
        Calculates the cost function (time taken) through the select pathway/route

        Args:
        - travel_edges: The list of edges of the selected route.
        - speed: The speed travel (constant) km/h
        - congestion_duration: The time taken for stuck in congestion (in minutes)
        - traffic_light_duration: The time taken for stuck in traffic light (in minutes)

        Return:
        - total_time (float): The total time taken to travel (in minutes)
        """

        total_time = ((self.get_edge_distance(travel_edges)/1000) / self.travel_speed) * 60 # km / (km/h) -> hrs -> mins

        if isinstance(travel_edges, str):
            travel_edges = [travel_edges]

        # time punishment for the route
        for i in range(len(travel_edges)):  # time punishment on a specific edge
            # congested edges
            if travel_edges[i] in self.congested_edges:
                total_time += self.congestion_duration[self.congested_edges.index(travel_edges[i])]

            # traffic light
            prev_node = self.decode_edge_to_node(travel_edges[i-1], direction = 'end') if i > 0 else ""
            current_node = self.decode_edge_to_node(travel_edges[i], direction = 'end')

            for index, search_nodes in enumerate(self.tl_nodes):
                if current_node in search_nodes and prev_node not in search_nodes:  # make sure you won't meet continual traffic light
                    total_time += self.tl_duration[index]

        return total_time


    # ------ Graph Visualisation ------
    def plot_visualised_result(self, travel_edges):
        """
        Plotting of network with selected route

        Args:
        - travel_edges (list): The list of edges of the selected route.
        - network_files_directory (str): The directory of the network files.
        - root_file (str): The initial name of the root file to be converted from network_file.

        Return:
        - Plot of network
        """

        nodes_dict = {}  # a list of x_coord and y_coord of every nodes
        for node in self.nodes:
            x_coord, y_coord = self.net.getNode(node).getCoord()
            nodes_dict[node] = (x_coord, y_coord)

        edges_dict = {}  # a list of from_point and to_point of every edges
        for edge in self.edges:
            from_id = self.net.getEdge(edge).getFromNode().getID()
            to_id = self.net.getEdge(edge).getToNode().getID()
            edges_dict[edge] = (from_id, to_id)

        # Draw the network layout
        net_G = nx.Graph()
        for edge in edges_dict:
            net_G.add_edge(edges_dict[edge][0], edges_dict[edge][1])
        pos = {node: nodes_dict[node] for node in nodes_dict}
        nx.draw(
            net_G, pos, with_labels=False,
            node_color='DimGray', node_size=10,
            edge_color='DarkGray'
        )

        # Draw the selected route
        route_G = nx.Graph()
        for edge in travel_edges:
            route_G.add_edge(edges_dict[edge][0], edges_dict[edge][1])
        nx.draw(
            route_G, pos, with_labels=False,
            node_color='SeaGreen', node_size=30,
            edge_color='MediumSeaGreen', width=3,

            arrows=True, arrowsize=7, arrowstyle='-|>'
        )

        # Draw traffic light nodes and congested edges in time evaluation
        if self.evaluation in ("time"):
            tl_lst = []  # light nodes
            for tl_node in self.tl_nodes:
                if isinstance(tl_node, list):
                    tl_lst.extend(tl_node)
                else:
                    tl_lst.append(tl_node)
            nx.draw_networkx_nodes(
                net_G, pos,
                nodelist=tl_lst, node_color='Crimson', node_size=30
            )

            congested_lst = [edges_dict[edge] for edge in self.congested_edges]  # congested edges
            nx.draw_networkx_edges(
                net_G, pos,
                edgelist=congested_lst, edge_color='IndianRed', width=3
            )

        plt.show()


    def plot_performance(self, num_episodes, logs):
        """
        Plotting of models' performance

        Args:
        - num_episodes (int): number of episodes it took for the model to converge, a trimmed one
        - logs (dict): the logs of the edges and states it took to converge

        Return:
        - Plot of the evaluation (time/distance) at each episode
        """

        plt.title("Performance of Agent")
        plt.xlabel("Episode")
        if self.evaluation in ("time"):
            plt.ylabel("Time")
            evaluation = [self.get_edge_time(logs[episode][1]) for episode in range(num_episodes)]
        else:
            plt.ylabel("Distance")
            evaluation = [self.get_edge_distance(logs[episode][1]) for episode in range(num_episodes)]
        plt.plot(range(num_episodes), evaluation)
        plt.show()
