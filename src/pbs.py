import time as timer
import heapq
from collections import deque
import random
import copy
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
from topological_sort import TopologyGraph
from cbs import detect_collisions_among_all_paths


def generate_priority_pairs(collision):

    priority_pairs = []

    ##############################
    # TODO Task 4.1: Generate list of priority pairs based on the given collision
    priority_pairs.append((collision['a1'], collision['a2']))
    priority_pairs.append((collision['a2'], collision['a1']))
    return priority_pairs

def get_lower_priority_agents(priority_pairs, agent):
    # Get the agents behind a given agent in a topological ordering
    tg = TopologyGraph(directed=True)
    tg.clear_graph()

    # construct graph
    for pair in priority_pairs:
        tg.Edge(pair[0], pair[1])

    if not tg.has_node(agent):
        return [agent]
    
    return tg.get_subsequent_nodes_in_topological_ordering(agent)

def get_higher_priority_agents(priority_pairs, agent):
    # Get the agents ahead of a given agent in a topological ordering
    tg = TopologyGraph(directed=True)
    tg.clear_graph()

    # construct graph
    for pair in priority_pairs:
        tg.Edge(pair[1], pair[0])

    if not tg.has_node(agent):
        return [agent]

    return tg.get_subsequent_nodes_in_topological_ordering(agent)

def collide_with_higher_priority_agents(node, agent):
    # Check if the given agent collides with any higher priority agents
    collisions = node['collisions']
    priority_pairs = node['priority_pairs']

    if collisions == [] or priority_pairs == []:
        return []

    higher_priority_agents = get_higher_priority_agents(node['priority_pairs'], agent)

    for collision in collisions:
        if collision['a1'] == agent and collision['a2'] in higher_priority_agents:
            return True
        elif collision['a2'] == agent and collision['a1'] in higher_priority_agents:
            return True

    return False 

class PBSSolver(object):
    """The high-level search of PBS."""

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
        self.search_stack = deque()

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node_to_stack(self, node):
        # Push node to search stack
        if len(self.search_stack) == 0:
            self.search_stack.append(node)
            return

        # index = 0
        index = len(self.search_stack)
        for i in range(len(self.search_stack)):
            if node['cost'] <= self.search_stack[i]['cost']:
                index = i
                break
        
        self.search_stack.insert(index, node)

    def pop_node_from_stack(self):
        # Pop node from search stack
        node = self.search_stack.popleft()
        return node

    def update_plan(self,node,i):
        # print(node['priority_pairs'])
        # Task 4.2 TODO : Refer to the given psuedocode or the cited paper for more details on what this function does
        agents = get_lower_priority_agents(node['priority_pairs'], i) # lower order
        priors = get_higher_priority_agents(node['priority_pairs'], i) # higher order
        priors = priors[1:] # remove i itself
        constraints = []
        time_factor = 5
        
        # print("i: ",i)
        # print("priors: ",priors)
        # print("agents: ",agents)
        
        # agents include i itself
        for (j,agent) in enumerate(agents):
            constraints = []
            # append last agent to priors
            if j>0:
                priors.append(agents[j-1])
            # append constraints for prior agents
            for prior in priors:
                max_step = (abs(self.goals[prior][0]-self.starts[prior][0])+abs(self.goals[prior][1]-self.starts[prior][1]))*time_factor
                for time_step in range(max_step):
                    if time_step >= len(node['paths'][prior]):
                        constraints.append({'agent': agent, 'loc': node['paths'][prior][-1], 'time_step': time_step})
                    else:
                        constraints.append({'agent': agent, 'loc': node['paths'][prior][time_step], 'time_step': time_step})
                # edge constraints
                for time_step in range(1, len(node['paths'][prior])):
                    constraints.append({'agent': agent, 'loc': [node['paths'][prior][time_step],node['paths'][prior][time_step-1]], 'time_step': time_step})\
                
            # for agent i, call low level a star search
            # print("agent: ",agent)
            # print("constraints: ",constraints)
            path = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent], agent, constraints)
            if path is None:
                return False
            # print("path: ",path)
            node['paths'][agent] = path
        return True


    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations

        """

        print('Start PBS')
        self.start_time = timer.time()

        # Generate the root node
        # priority_pairs   - list of priority pairs
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        
        ##############################
        # Task 4.2: Initialize the root node dict, what will be the initial priority pairs for standard PBS?
        #
        # TODO      
        # root_priority_pairs = []
        # root priority is in the order of agents number
        # for i in range(self.num_of_agents):
        #     for j in range(i+1, self.num_of_agents):
        #         root_priority_pairs.append((i,j))
        root = {'cost': 0,
                'priority_pairs': [],
                'collisions': [],
                'paths': [[] for i in range(self.num_of_agents)]
                }

        ##############################
        # Task 4.2: Find initial path for each agent
        #   
        for i in range(self.num_of_agents):  
            self.update_plan(root,i)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions_among_all_paths(root['paths'])
        
        ##############################
        # Task 4.2: Add root to search stack
        # TODO
        self.push_node_to_stack(root)


        while len(self.search_stack)>0:

            ##############################
            # Task 4.2: Get next node from stack
            #     
            # TODO
            next_node = self.pop_node_from_stack()

            # print expanded node info
            # print("Expanded node cost: {} priority {} collisions {}".format(next_node['cost'],(next_node['priority_pairs']),(next_node['collisions'])))
            # for p in next_node['paths']:
            #     print(p)
            if len(next_node['collisions']) == 0:
                self.print_results(next_node)
                return next_node['paths']

            collision = next_node['collisions'][0]
            ##############################
            # Task 4.1: Generate priority pairs for this collision
            #     
            priority_pairs = generate_priority_pairs(collision)
            
            # Create child nodes
            for priority_pair in priority_pairs:
                #agent = constraint['agent']

                # Create new child node
                child = copy.deepcopy(next_node)

                ##############################
                # Task 4.2:  
                # - Duplicate detection - check if priority pair already exists in parent node
                # - If priority pair already exists in parent node, skip this child
                # - Else add priority pair to child priority pairs
                # TODO
                if priority_pair in child['priority_pairs']:
                    continue
                else:
                    child['priority_pairs'].append(priority_pair)

                ##############################
                # Task 4.2:  Replan for all agents in topological order
                #     
                # print("node: ")
                # print(next_node)
                update_success = self.update_plan(child,priority_pair[1]) # 0 has higher priority, it does not change path
                if update_success:
                    child['cost'] = get_sum_of_cost(child['paths'])
                    child['collisions'] = detect_collisions_among_all_paths(child['paths'])
                    # print("generated child cost: {} priority {} collisions {}".format(child['cost'],(child['priority_pairs']),(child['collisions'])))
                else:
                    print("No solution found for child")
                    continue
                self.push_node_to_stack(child)
            # import pdb;pdb.set_trace()
            ##############################
            # Task 4.2:  # Add nodes to stack from heap in non increasing order of cost
            # TODO
            # self.push_node_to_stack(child)
        import pdb;pdb.set_trace()
        return None


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))