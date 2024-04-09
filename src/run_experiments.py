#!/usr/bin/python
import os
import rospy
import argparse
import glob
from pathlib import Path
from cbs import CBSSolver
from pbs import PBSSolver
# from independent import IndependentSolver
# from joint_state import JointStateSolver
# from prioritized import PrioritizedPlanningSolver
from visualize import Animation
from single_agent_planner import get_sum_of_cost

SOLVER = "CBS"

def print_mapf_instance(my_map, starts, goals):
    rospy.loginfo('Start locations')
    print_locations(my_map, starts)
    rospy.loginfo('Goal locations')
    print_locations(my_map, goals)


def print_locations(my_map, locations):
    starts_map = [[-1 for _ in range(len(my_map[0]))] for _ in range(len(my_map))]
    for i in range(len(locations)):
        starts_map[locations[i][0]][locations[i][1]] = i
    to_print = ''
    for x in range(len(my_map)):
        for y in range(len(my_map[0])):
            if starts_map[x][y] >= 0:
                to_print += str(starts_map[x][y]) + ' '
            elif my_map[x][y]:
                to_print += '@ '
            else:
                to_print += '. '
        to_print += '\n'
    rospy.loginfo(to_print)


def import_mapf_instance(filename):
    f = Path(filename)
    if not f.is_file():
        raise BaseException(filename + " does not exist.")
    f = open(filename, 'r')
    # first line: #rows #columns
    line = f.readline()
    rows, columns = [int(x) for x in line.split(' ')]
    rows = int(rows)
    columns = int(columns)
    # #rows lines with the map
    my_map = []
    for r in range(rows):
        line = f.readline()
        my_map.append([])
        for cell in line:
            if cell == '@':
                my_map[-1].append(True)
            elif cell == '.':
                my_map[-1].append(False)
    # #agents
    line = f.readline()
    num_agents = int(line)
    # #agents lines with the start/goal positions
    starts = []
    goals = []
    for a in range(num_agents):
        line = f.readline()
        sx, sy, gx, gy = [int(x) for x in line.split(' ')]
        starts.append((sx, sy))
        goals.append((gx, gy))
    f.close()
    return my_map, starts, goals


def main():
    rospy.init_node('mapf_exploration_node', anonymous=True)
    solver = rospy.get_param('~solver', SOLVER)
    instance = rospy.get_param('~instance', 'instances/test_21.txt')
    batch = rospy.get_param('~batch', False)

    rospy.loginfo(f"Instance Path received: {instance}")
    rospy.loginfo(f"Solver argument received: {solver}")
    result_file = open("results.csv", "w", buffering=1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(script_dir, instance)
    
    if not os.path.exists(file):
        rospy.logerr(f"Instance file '{file}' does not exist.")
        return
    
    print("***Import an instance***")
    my_map, starts, goals = import_mapf_instance(file)
    print_mapf_instance(my_map, starts, goals)

    if solver == "CBS":
        print("***Run CBS***")
        cbs = CBSSolver(my_map, starts)
        paths = cbs.explore_environment()
    elif solver == "PBS":
        rospy.loginfo("***Run PBS***")
        solver = PBSSolver(my_map, starts, goals)
        paths = solver.find_solution()
    else:
        raise RuntimeError("Unknown solver!")

    cost = get_sum_of_cost(paths)
    result_file.write("{},{}\n".format(file, cost))

    if not batch:
        print("***Test paths on a simulation***")
        animation = Animation(my_map, starts, goals, paths)
        animation.show()

    result_file.close()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
