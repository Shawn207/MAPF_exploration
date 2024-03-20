import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
import copy

def detect_first_collision_for_path_pair(path1, path2):
    ##############################
    # Task 2.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.

    # determine longer path
    longer_path = None
    shorter_path = None
    if len(path1) > len(path2):
        longer_path = path1
        shorter_path = path2
    else:
        longer_path = path2
        shorter_path = path1
    
    time_step = 0
    while time_step < len(longer_path):
        # vertex collision
        if time_step < len(shorter_path):
            if get_location(path1, time_step) == get_location(path2, time_step):
                return {'location': get_location(path1, time_step), 'time_stamp': time_step}
        else:
            if get_location(longer_path, time_step) == get_location(shorter_path, len(shorter_path) - 1):
                return {'location': get_location(longer_path, time_step), 'time_stamp': time_step}
            
        # edge collision
        if get_location(path1, time_step) == get_location(path2, time_step - 1) and get_location(path1, time_step - 1) == get_location(path2, time_step):
            return {'location': [get_location(path1, time_step), get_location(path1, time_step - 1)], 'time_stamp': time_step}
        
        time_step += 1
    
    return None


def detect_collisions_among_all_paths(paths):
    ##############################
    # Task 2.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    collisions = []
    for i in range(len(paths) - 1):
        for j in range(i + 1, len(paths)):
            collision = detect_first_collision_for_path_pair(paths[i], paths[j])
            if collision is not None:
                collision['a1'] = i
                collision['a2'] = j
                collisions.append(collision)
    
    return collisions


def standard_splitting(collision):
    ##############################
    # Task 2.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep

    constraints = []
    constraints.append({'agent': collision['a1'], 'loc': collision['location'], 'time_step': collision['time_stamp']})
    constraints.append({'agent': collision['a2'], 'loc': collision['location'], 'time_step': collision['time_stamp']})
    
    return constraints




class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []
        self.information_map = copy.deepcopy(1-my_map)
        for i in range(len(self.information_map)):
            for j in range(len(self.information_map[0])):
                self.information_map[i][j] = 1 - self.information_map[i][j]


        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations

        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions_among_all_paths(root['paths'])
        self.push_node(root)

        # Task 2.1: Testing
        # print(root['collisions'])

        # Task 2.2: Testing
        # for collision in root['collisions']:
        #     print(standard_splitting(collision))

        ##############################
        # Task 2.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit
        while self.open_list:
            # print("pop a node")
            curr = self.pop_node()
            # print(curr)
            # print(curr['cost'])
            # return solution if this node has no collision
            if len(curr['collisions']) == 0:
                # import pdb;pdb.set_trace()
                return curr['paths']
            
            # choose the first collision and convert to a list of constraints
            collision = curr['collisions'][0]
            constraints = standard_splitting(collision)
            
            # Add a new child node to the open list for each constraint
            for constraint in constraints:
                # print("Add a new child node to the open list for each constraint")
                new_node = copy.deepcopy(curr)
                if constraint not in new_node['constraints']:
                    new_node['constraints'].append(constraint)
                    new_path = a_star(self.my_map, self.starts[constraint['agent']], self.goals[constraint['agent']], self.heuristics[constraint['agent']], constraint['agent'], new_node['constraints'])
                if new_path is None:
                    continue
                new_node['paths'][constraint['agent']] = new_path
                new_node['cost'] = get_sum_of_cost(new_node['paths'])
                new_node['collisions'] = detect_collisions_among_all_paths(new_node['paths'])
                # print(constraint)
                # print(new_path)
                # print(new_node['collisions'])
                self.push_node(new_node)
            # import pdb;pdb.set_trace()
            
        # update infomration map for each path
        for path in root['paths']:
            for i in range(len(path)):
                self.information_map[path[i][0]][path[i][1]] = 0

        # These are just to print debug output - can be modified once you implement the high-level search
        # self.print_results(root)
        # import pdb;pdb.set_trace()
        return root['paths']


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
