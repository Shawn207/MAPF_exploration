#!/usr/bin/python
import os
import rospy
import argparse
import glob
from pathlib import Path
from cbs import CBSSolver
from pbs import PBSSolver
from single_agent_planner import get_sum_of_cost
from mapf_exploration.msg import robotStates
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as waypoints
from mapf_exploration.msg import PathArray 
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import math

class MAPF_Publisher:
    def __init__(self, solver: str, instance: str, batch: bool, robotState_topic: str, pub_path_topic: str):
        """
        :param 
        """

        rospy.loginfo(f"Instance Path received: {instance}")
        rospy.loginfo(f"Solver argument received: {solver}")
        self.instance = instance
        self.solver = solver
        self.batch = batch
        self.replan = False

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(script_dir, instance)
        
        if not os.path.exists(file):
            raise FileExistsError(f"Instance file '{file}' does not exist.")
        else:
            self.file = file
    
        print("***Import an instance***")
        self.my_map, self.starts, self.goals = self.import_mapf_instance(file)
        self.print_mapf_instance(self.my_map, self.starts, self.goals)

        rospy.loginfo("MAPF initialization complete. Ready to start planning")

        self.path_publisher = rospy.Publisher(pub_path_topic, PathArray, queue_size=100)
        self.pose_subscriber = rospy.Subscriber(robotState_topic, robotStates, self.robotStateCB)
        self.timer = rospy.Timer(rospy.Duration(0.03), self.solve)
        print("current time:", rospy.Time.now())
    def robotStateCB(self, robotState_msg: robotStates):
        self.robotState = robotState_msg
        print("robot state")
        print(self.robotState.robot_id)
        print(self.robotState.position[0])
        print(self.robotState.position[1])
        self.starts[self.robotState.robot_id-1] = (int(self.robotState.position[0]), int(self.robotState.position[1]))
        self.replan = True

    def solve(self, event=None): 
        """ callback function for publisher """
        # rospy.loginfo("Solver callback triggered")
        if self.replan:
            rospy.loginfo("Solver callback triggered")
            result_file = open("results.csv", "w", buffering=1)

            if self.solver == "CBS":
                print("***Run CBS***")
                cbs = CBSSolver(self.my_map, self.starts)
                paths = cbs.explore_environment()
                path_array = PathArray()
                for path in paths:
                    path_waypoints = waypoints()
                    for waypoint in path:
                        point = PoseStamped()
                        print(waypoint)
                        point.pose.position.x = waypoint[0]
                        point.pose.position.y = waypoint[1]
                        path_waypoints.poses.append(point)
                    path_array.PathArray.append(path_waypoints)
                # path_array.header.stamp = img_timestamp        
                self.path_publisher.publish(path_array)
            else:
                raise RuntimeError("Unknown solver!")

            cost = get_sum_of_cost(paths)
            result_file.write("{},{}\n".format(self.file, cost))

            if not self.batch:
                print("***Test paths on a simulation***")
                # animation = Animation(self.my_map, self.starts, self.goals, paths)
                # animation.show()

            result_file.close()
            self.replan = False

    def print_mapf_instance(self, my_map, starts, goals):
        rospy.loginfo('Start locations')
        self.print_locations(my_map, starts)
        rospy.loginfo('Goal locations')
        self.print_locations(my_map, goals)


    def print_locations(self, my_map, locations):
        starts_map = [[-1 for _ in range(len(my_map[0]))] for _ in range(len(my_map))]
        for i in range(len(locations)):
            starts_map[locations[i][0]][locations[i][1]] = i
        to_print = '\n'
        to_print += ''
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


    def import_mapf_instance(self, filename):
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

if __name__ == '__main__':
    SOLVER = "CBS"

    rospy.init_node('mapf_exploration_node', anonymous=True)
    solver = rospy.get_param('~solver', SOLVER)
    instance = rospy.get_param('~instance', 'instances/test_21.txt')
    batch = rospy.get_param('~batch', False)
    robotState_topic = '/dynamicExploration/robotState'
    pub_path_topic = 'multi_waypoints'


    publisher = MAPF_Publisher(solver, instance, batch, robotState_topic, pub_path_topic)

    rospy.spin()
