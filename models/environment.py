import sys
import sumolib
import math
import networkx as nx
import matplotlib.pyplot as plt
import random

class traffic_env:
    def __init__ (self, network_file, tls, congestion = [], evaluation = "", congestion_level = ""):
        # 1. Define network_file
        self.network_file = network_file  # read the file

        self.net = sumolib.net.readNet(network_file)  # file -> net
        self.nodes = [node.getID().upper() for node in self.net.getNodes()]  # net -> nodes (ID)
        self.edges = [edge.getID() for edge in self.net.getEdges()]  # net -> edges (ID)

        self.tls = tls  # [tl_id][link_index]=[90] (dict)
        self.tls_space = [tl.getID() for tl in self.net.getTrafficLights()]
        self.tls_meet = []  # to print on map
        self.congestion_meet = []  # to print on map

        self.action_space = [0, 1, 2, 3]  # action_space
        self.state_space = self.nodes  # state_space
        self.edge_label = self.decode_edges_to_label()  # give every edges a label by their direction in the aspect of x-y coordinate


        # 2. Define congestions edges with its original pattern
        if congestion:  # if congestion is defined
            self.congested_edges = [item[0] for item in congestion]
            self.congestion_duration = [item[1] for item in congestion]  # the duration of so called "traffic jam"

            for edge in self.congested_edges:  # make sure that all congested_edges are in the net
                if edge not in self.edges:
                    sys.exit(f'Error: Invalid congestion_edges {edge}')
            # print(f'Congested Edges: {list(zip(self.congested_edges, self.congestion_duration))}')
            # print(f'Congested/Total: {len(self.congested_edges)}/{len(self.edges)}')

        else:  # if congestion is not defined, then set edges and its duration randomly
            if congestion_level == "low":
                traffic_level = 0.05  # 5% congested
            elif congestion_level == "medium":
                traffic_level = 0.10  # 10% congested
            elif congestion_level == "high":
                traffic_level = 0.20  # 20% congested
            self.congested_edges = random.sample(self.edges, round(len(self.edges) * traffic_level))
            self.congestion_duration = [random.randint(60, 120) for _ in range(len(self.congested_edges))]  # 1~2 min
            # print(f'Congested Edges: {list(zip(self.congested_edges, self.congestion_duration))}')
            # print(f'Congested/Total: {len(self.congested_edges)}/{len(self.edges)}')


        # 3. Define evaluation type
        if evaluation not in ('distance', 'time'):
            sys.exit('Error: Invalid evaluation type, provide only "distance" or "time"')
        self.evaluation = evaluation


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
                sys.exit(f'Error: Edge {edge} not in Edges Space ...call by get_edge_distance')
            # Sum up the distance of each edges
            total_distance += self.net.getEdge(edge).getLength()

        return total_distance


    # Find the total time taken by giving a single edge / an edge_path
    def get_edge_time(self, travel_edges):
        """
        Calculates the cost function (time taken) through the select pathway/route

        Args:
        - travel_edges: The list of edges of the selected route.
        - congestion_duration: The time taken for stuck in congestion (in seconds)

        Return:
        - total_time (float): The total time taken to travel (in seconds)
        """

        if isinstance(travel_edges, str):
            travel_edges = [travel_edges]

        total_time = 0
        for edge in travel_edges:
            # Check if edges are in the edges list
            if edge not in self.edges:
                sys.exit(f'Error: Edge {edge} not in Edges Space ...call by get_edge_time')
            # Sum up the distance of each edges
            total_time += self.net.getEdge(edge).getLength() / self.net.getEdge(edge).getSpeed()

        # time punishment for the route
        for i in range(len(travel_edges)):  # time punishment on a specific edge because of only congested edges
            if travel_edges[i] in self.congested_edges:
                total_time += self.congestion_duration[self.congested_edges.index(travel_edges[i])]

        return total_time


    # Find the time offset caused by the traffic light
    def get_tl_offset(self, travel_edges):
        """
        Calculates the time offset caused by the traffic light

        Args:
        - travel_edges: The list of edges of the selected route.
        - tls: The dictionary of tls data

        Return:
        - (current_time - self.get_edge_time(travel_edges)) (float): The time offset caused by the traffic light
            where current_time: the time taken to travel the selected route
                  get_edge_time(travel_edges): the time taken to travel the selected route w/o considering traffic light
        """
        self.tls_meet = []  # to print on map
        self.congestion_meet = []  # to print on map

        if isinstance(travel_edges, str):
            travel_edges = [travel_edges]

        current_time = 0
        for edge in range(len(travel_edges) - 1):
            current_edge = travel_edges[edge]
            next_edge = travel_edges[edge+1]

            # 0. Check if edges are in the edges list
            if current_edge not in self.edges:
                sys.exit(f'Error: Edge {current_edge} not in Edges Space ...call by get_tl_offset')

            if current_edge in self.congested_edges and current_edge not in self.congestion_meet:
                self.congestion_meet.append(current_edge)  # to print on map

            # 1. Sum up the distance of each edges
            current_time += self.net.getEdge(current_edge).getLength() / self.net.getEdge(current_edge).getSpeed()

            # 2. Find the end point of the edge
            tl = self.net.getEdge(current_edge).getToNode().getID()
            if tl not in self.tls_space:
                continue

            self.tls_meet.append(tl)  # to print on map

            # 3. Find the connection of current_edge and next_edge
            connections_set = self.net.getTLS(tl).getConnections()

            for j in range(len(connections_set)):
                connection = connections_set[j]
                if connection[0] in self.net.getEdge(current_edge).getLanes() and connection[1] in self.net.getEdge(next_edge).getLanes():
                    break

            # 4. Derive that this connection is the nth link of this tl_node
            tl_phase = self.tls[tl][connection[2]]

            idle_time = 0
            if tl_phase[int(current_time) % 90] != "r":
                continue
            else:
                for phase_index in range(90):
                    if tl_phase[(int(current_time) + phase_index) % 90] != "r":
                        idle_time = phase_index
                        break

            # 5. Sum up the idle
            current_time += idle_time

        return current_time - self.get_edge_time(travel_edges)


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
            nx.draw_networkx_nodes(
                net_G, pos,
                nodelist=self.tls_meet, node_color='Crimson', node_size=30
            )

            congested_lst = [edges_dict[edge] for edge in self.congested_edges]  # congested edges
            nx.draw_networkx_edges(
                net_G, pos,
                edgelist=congested_lst, edge_color='Gold', width=3
            )

            congestion_meet = [edges_dict[edge] for edge in self.congestion_meet]  # congested edges
            nx.draw_networkx_edges(
                net_G, pos,
                edgelist=congestion_meet, edge_color='IndianRed', width=3
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
            evaluation = [(self.get_edge_time(logs[episode][1]) + self.get_tl_offset(logs[episode][1]))/60  for episode in range(num_episodes)]
        else:
            plt.ylabel("Distance")
            evaluation = [self.get_edge_distance(logs[episode][1]) for episode in range(num_episodes)]
        plt.plot(range(num_episodes), evaluation)
        plt.show()
