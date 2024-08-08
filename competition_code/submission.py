"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
    
    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )


    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        
        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )
        # Calculate dynamic lookahead based on velocity
        lookahead_distance = int(vehicle_velocity_norm // 5)  # Example: Look ahead more as speed increases

        # Ensure lookahead distance stays within reasonable bounds
        lookahead_distance = np.clip(lookahead_distance, 10, 30)  # Example: between 1 and 10 waypoints

        # Adjust target waypoint based on current velocity
        waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + lookahead_distance) % len(self.maneuverable_waypoints)]
        curve_ahead = self.maneuverable_waypoints[(self.current_waypoint_idx + 70) % len(self.maneuverable_waypoints)]

        # Calculate delta vector towards the target waypoint
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        curve_vector_to_waypoint = (curve_ahead.location - vehicle_location)[:2]
        curve_heading_to_waypoint = np.arctan2(curve_vector_to_waypoint[1],curve_vector_to_waypoint[0])

        # Calculate delta angle towards the target waypoint
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])
        curve_delta_heading = normalize_rad(curve_heading_to_waypoint - vehicle_rotation[2])

        # Proportional controller to steer the vehicle towards the target waypoint
        steer_control = (
            -8.0 / np.sqrt(vehicle_velocity_norm) * delta_heading / np.pi
        ) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)

        # Proportional controller to control the vehicle's speed towards 40 m/s
        throttle_control = 0.05 * (85 - vehicle_velocity_norm)

        curvature = np.pi - delta_heading
        curve_ahead_curvature = np.pi - curve_delta_heading

        print("curvature", curvature, "speed:", vehicle_velocity_norm)
        print("throttle:", throttle_control)

        # If the curvature is high, slow down; if it's low, speed up
        if 2700 > self.current_waypoint_idx > 2660:
            throttle_control = min(0.0, throttle_control - 0.05)
        else:
            if curvature > np.pi / 4:  # Example threshold for turns
                throttle_control = max(0.0, throttle_control - 1)  # Reduce throttle
            elif np.pi/6 < curvature < np.pi / 4:
                throttle_control = max(0.0, throttle_control + 1)
            elif 0 == curvature or 0 < curvature < np.pi/6:
                throttle_control = 1
            if curve_ahead_curvature > np.pi / 3:
                throttle_control = max(0.0, throttle_control - 1)

        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": steer_control,
            "brake": np.clip(-throttle_control, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
        return control
