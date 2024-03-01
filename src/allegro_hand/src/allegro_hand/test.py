#!/usr/bin/env python

import sys
import argparse
import numpy as np
import rospy
import torch
import torch.nn as nn
import torch.optim as optim
from liballegro import AllegroClient
"""
This file contains several examples of the AllegroClient python library that
allows you to interact with the Allegro hand directly using python.

Set the allegro hand topic directly using:
   --hand_prefix=allegroHand_0
(or some other topic name.)

"""

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(1,10)
        self.fc2 = nn.Linear(10,1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def test(allegro_client):
    position = np.zeros(16)
    position[[0, 4, 8, 12]] = 0.0
    hz = 4
    r = rospy.Rate(hz)
    position = np.array([-0.1194, 1.2068, 1.0, 1.4042,-0.0093, 1.2481, 1.4073, 0.8163,0.1116, 1.2712, 1.3881, 1.0122,0.6017, 0.2976, 0.9034, 0.7929])
    allegro_client.command_joint_position(position)
    rospy.sleep(3)
    position = np.array([-0.1220, 0.4, 0.6, -0.0769,0.0312, 0.4, 0.6, -0.0,0.1767, 0.4, 0.6, -0.0528,0.5284, 0.3693, 0.8977, 0.4863])
    allegro_client.command_joint_position(position)
    rospy.sleep(3)
    position = np.array([0.0885, 0.4, 0.6, -0.0704,0.0312, 0.4, 0.6, -0.0,0.1019, 1.2375, 1.1346,1.0244, 1.0, 0.6331, 1.3509, 1.0])
    allegro_client.command_joint_position(position)
    rospy.sleep(3)
    #position = np.array([-0.1194, 1.2068, 1.0, 1.4042,0.0312, 0.4, 0.6, -0.0,0.1116, 1.2712, 1.3881, 1.0122,0.6017, 0.2976, 0.9034, 0.7929])
    #allegro_client.command_joint_position(position)
    #allegro_client.command_joint_position(position)
    r.sleep()
    return
    
def test2(allegro_client):

    hz = 4
    r = rospy.Rate(hz)

    position = np.array([-0.1194, 1.2068, 1.0, 1.4042,-0.0093, 1.2481, 1.4073, 0.8163,0.1116, 1.2712, 1.3881, 1.0122,0.6017, 0.2976, 0.9034, 0.7929])
    allegro_client.command_joint_position(position)
    rospy.sleep(3)

    model = SimpleClassifier()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    X_train = torch.tensor([[0.5],[1.5],[2.5],[3.5]], dtype=torch.float32)
    y_train = torch.tensor([[0], [1], [1], [1]], dtype=torch.float32)

    epochs = 2000
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        test_input = torch.tensor([[0.2], [1.3], [0.5]], dtype=torch.float32)
        predictions = model(test_input)
        for i, pred in enumerate(predictions):
            print(f'Input: {test_input[i][0]}, Predicted: {pred.item()}, Class: {"<1" if pred.item() < 0.5 else ">=1"}')
            if pred.item() < 0.5:
                allegro_client.command_hand_configuration('ready')
                rospy.sleep(3)
            else:
                allegro_client.command_hand_configuration('index_pinch')
                rospy.sleep(3)
    #position = np.array([0.0885, 0.4, 0.6, -0.0704,0.0312, 0.4, 0.6, -0.0,0.1019, 1.2375, 1.1346,1.0244, 1.0, 0.6331, 1.3509, 1.0])
    #allegro_client.command_joint_position(position)
    #rospy.sleep(3)


    


def run(args):

    parser = argparse.ArgumentParser(description='Allegro python library')
    parser.add_argument('--hand_prefix', type=str,
                        help='ROS topic prefix for the hand.',
                        default='allegroHand')

    (parsed_args, other_args) = parser.parse_known_args(args)
    rospy.init_node('example_allegro_lib', other_args)

    client = AllegroClient(hand_topic_prefix=parsed_args.hand_prefix)
    rospy.sleep(0.5)  # Wait for connections.

    rospy.loginfo('== Commanding hand configuration: home... ==')
    client.command_hand_configuration('home')

 
    client.command_hand_configuration('home')


    rospy.loginfo('== test script ==')
    test2(client)


if __name__ == '__main__':
    args = sys.argv[1:]
    run(args)
