import time

import os
import pickle
import numpy as np
import sympy
import logging
from typing import Tuple
import constants

from shapely import geometry
import pdb
from collections import defaultdict
import shapely.geometry
import math


def get_distance(point1, point2):
    return math.sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2))


class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


class Player:
    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger, golf_map: sympy.Polygon, start: sympy.geometry.Point2D, target: sympy.geometry.Point2D, map_path: str, precomp_dir: str) -> None:
        """Initialise the player with given skill.

        Args:
            skill (int): skill of your player
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
            logger (logging.Logger): logger use this like logger.info("message")
            golf_map (sympy.Polygon): Golf Map polygon
            start (sympy.geometry.Point2D): Start location
            target (sympy.geometry.Point2D): Target location
            map_path (str): File path to map
            precomp_dir (str): Directory path to store/load precomputation
        """
        # # if depends on skill
        # precomp_path = os.path.join(precomp_dir, "{}_skill-{}.pkl".format(map_path, skill))
        # # if doesn't depend on skill
        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))
        
        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)
        self.skill = skill
        self.rng = rng
        self.logger = logger
        # TODO: define the risk rate, which means we can only take risks at 10%
        self.risk = 0.1
        self.simulate_times = 100
        self.tolerant_times = self.simulate_times * self.risk
        self.remember_middle_points = []

        self.turn = 0
        self.shapely_golf_map = None
        self.point_dict = defaultdict(int)

        self.last_result = None

    def water_boolean(self, poly, grid_points):
        water_grid = []

        for i, row in enumerate(grid_points):
            water_grid.append([])
            for j, point in enumerate(row):
                thebool = poly.contains(point)

                water_grid[i].append(thebool)

        return np.array(water_grid)

    def make_grid(self, golf_map: sympy.Polygon, target: sympy.geometry.Point2D,
                  curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D):

        poly = geometry.Polygon([p.x, p.y] for p in golf_map.vertices)
        target_shapely = geometry.Point(target[0], target[1])

        (xmin, ymin, xmax, ymax) = golf_map.bounds
        list_of_lists = []
        list_of_distances = []

        queue = []
        dimension = 20
        allowed_distance = (constants.max_dist + self.skill) / (1. + constants.extra_roll)
        threshold = 20.0
        amt = 1
        grid_of_scores = np.array(np.ones((dimension, dimension)) * 100)

        xmin = float(xmin)
        ymin = float(ymin)
        xmax = float(xmax)
        ymax = float(ymax)

        xcoords, ycoords = np.meshgrid(np.linspace(xmin, xmax, dimension), np.linspace(ymin, ymax, dimension))

        for x_index in range(len(xcoords)):
            list_of_lists.append([])
            list_of_distances.append([])
            for y_index in range(len(ycoords)):
                # considered_point = sympy.geometry.Point2D(xcoords[y_index,x_index],ycoords[y_index,x_index])
                considered_point = geometry.Point(xcoords[y_index, x_index], ycoords[y_index, x_index])
                list_of_lists[x_index].append(considered_point)

        grid_of_scores = self.real_bfs(poly, list_of_lists, target_shapely, allowed_distance, grid_of_scores)

        return grid_of_scores, list_of_lists

    def real_bfs(self, poly, list_of_lists, target_shapely, allowed_distance, grid_of_scores):
        queue = []
        water_grid = self.water_boolean(poly, list_of_lists)  # True if on LAND
        for x_index in range(len(list_of_lists)):
            for y_index in range(len(list_of_lists[0])):
                thedistance = target_shapely.distance(list_of_lists[x_index][y_index])
                if (thedistance < allowed_distance) and water_grid[x_index][y_index]:
                    queue.append((x_index, y_index))
                    grid_of_scores[x_index][y_index] = 1

        while (len(queue) != 0):

            elem = queue.pop()
            # get all points that are < distance from elem
            elem_score = grid_of_scores[elem[0]][elem[1]]
            points_to_consider = []
            for x_index in range(len(list_of_lists)):
                for y_index in range(len(list_of_lists[0])):
                    elem_point = list_of_lists[elem[0]][elem[1]]
                    distance = elem_point.distance(list_of_lists[x_index][y_index])
                    if distance < allowed_distance:
                        points_to_consider.append((x_index, y_index))
            for point in points_to_consider:
                if water_grid[point[0], point[1]]:
                    (x_index, y_index) = point
                    if elem_score + 1 < grid_of_scores[x_index][y_index]:
                        grid_of_scores[x_index][y_index] = elem_score + 1
                        queue.append(point)
        return grid_of_scores

    def play(self, score: int, golf_map: sympy.Polygon, target: sympy.geometry.Point2D,
             curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D,
             prev_landing_point: sympy.geometry.Point2D, prev_admissible: bool) -> Tuple[float, float]:
        """Function which based n current game state returns the distance and angle, the shot must be played 

        Args:
            score (int): Your total score including current turn
            golf_map (sympy.Polygon): Golf Map polygon
            target (sympy.geometry.Point2D): Target location
            curr_loc (sympy.geometry.Point2D): Your current location
            prev_loc (sympy.geometry.Point2D): Your previous location. If you haven't played previously then None
            prev_landing_point (sympy.geometry.Point2D): Your previous shot landing location. If you haven't played previously then None
            prev_admissible (bool): Boolean stating if your previous shot was within the polygon limits. If you haven't played previously then None

        Returns:
            Tuple[float, float]: Return a tuple of distance and angle in radians to play the shot
        """

        if self.turn == 0:
            grid_scores, point_map = self.make_grid(golf_map, target, curr_loc, prev_loc)
            for x_index in range(len(point_map)):
                for y_index in range(len(point_map[0])):
                    if int(grid_scores[x_index][y_index]) != 100:
                        self.point_dict[Point(point_map[x_index][y_index].x, point_map[x_index][y_index].y)] = int(
                            grid_scores[x_index][y_index])
            self.shapely_golf_map = shapely.geometry.polygon.Polygon(golf_map.vertices)

        self.turn += 1

        if not prev_admissible and self.last_result is not None:
            return self.last_result

        # 1. always try greedy first
        required_dist = curr_loc.distance(target)
        roll_factor = 1. + constants.extra_roll
        if required_dist < constants.min_putter_dist:
            roll_factor = 1.0
        distance = sympy.Min(constants.max_dist + self.skill, required_dist / roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)

        is_greedy = True
        failed_times = 0
        # simulate the actual situation to ensure fail times will not be larger than self.tolerant_times
        for _ in range(self.simulate_times):
            is_succ, _ = self.simulate_shapely_once(distance, angle, curr_loc, self.shapely_golf_map)
            if not is_succ:
                failed_times += 1
                if failed_times > self.tolerant_times:
                    is_greedy = False
                    break
        if is_greedy:
            self.logger.info(str(self.turn) + "select greedy strategy to go")
            self.last_result = (distance, angle)
            return (distance, angle)

        self.logger.info(str(self.turn) + "sample points to go")
        desire_distance, desire_angle = self.get_safe_sample_points_inside_circle(self.point_dict, curr_loc, distance, target, golf_map, prev_admissible)
        self.last_result = (desire_distance, desire_angle)
        return (desire_distance, desire_angle)

    def simulate_shapely_once(self, distance, angle, curr_loc, golf_map):
        actual_distance = self.rng.normal(distance, distance / self.skill)
        actual_angle = self.rng.normal(angle, 1 / (2 * self.skill))

        # landing_point means the land point(the golf can skip for a little distance),
        # final_point means the final stopped point, it is not equal
        if distance < constants.min_putter_dist:
            landing_point = shapely.geometry.Point(curr_loc.x, curr_loc.y)
            final_point = shapely.geometry.Point(curr_loc.x + actual_distance * np.cos(actual_angle),
                                                 curr_loc.y + actual_distance * np.sin(actual_angle))

        else:
            landing_point = shapely.geometry.Point(curr_loc.x + actual_distance * np.cos(actual_angle),
                                                   curr_loc.y + actual_distance * np.sin(actual_angle))
            final_point = shapely.geometry.Point(
                curr_loc.x + (1. + constants.extra_roll) * actual_distance * np.cos(actual_angle),
                curr_loc.y + (1. + constants.extra_roll) * actual_distance * np.sin(actual_angle))

        is_inside = golf_map.contains(landing_point) and golf_map.contains(final_point)

        return is_inside, final_point

    # points_score --> dictionary of (point, score)
    def get_safe_sample_points_inside_circle(self, points_score, curr_loc, radius, target, golf_map, prev_admissible):
        circle_points = dict()
        max_dist = radius

        for points in points_score.keys():
            if get_distance(curr_loc, points) <= max_dist:
                circle_points[points] = points_score[points]

        sorted_points_score = dict(sorted(circle_points.items(), key=lambda x: x[1]))
        smallest_score = min(sorted_points_score.values())
        smallest_score_points = dict()
        for points, value in sorted_points_score.items():
            if value == smallest_score:
                smallest_score_points[points] = get_distance(target, points)

        closest2target_points = dict(sorted(smallest_score_points.items(), key=lambda x: x[1]))
        safe_point = None
        unsafe_points2score = dict()
        for point in closest2target_points.keys():
            succ_times = 0
            for _ in range(self.simulate_times):
                angle = sympy.atan2(point.y - curr_loc.y, point.x - curr_loc.x)
                is_succ, _ = self.simulate_shapely_once(get_distance(curr_loc, point), angle, curr_loc,
                                                        self.shapely_golf_map)
                succ_times += is_succ

            if succ_times / self.simulate_times >= 1 - self.risk:
                safe_point = point
                break

            unsafe_points2score[point] = succ_times

        if safe_point is None:
            unsafe_points = sorted(unsafe_points2score.items(), key=lambda x: -x[1])
            safe_point = unsafe_points[0][0]
            safe_point_max_succ_times = unsafe_points[0][1]
            quadrant = (safe_point.y - curr_loc.y, safe_point.x - curr_loc.x)
            desire_angle = sympy.atan2(safe_point.y - curr_loc.y, safe_point.x - curr_loc.x)
            self.logger.info(str(self.turn) + "reach unsafe state")

            # try to go to middle points
            angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
            roll_factor = 1. + constants.extra_roll
            desire_distance, desire_angle, max_succ_times = self.go_for_middle_points_in_circle(prev_admissible, curr_loc, radius, angle, golf_map, quadrant, desire_angle, roll_factor)
            if max_succ_times is None or safe_point_max_succ_times > max_succ_times:
                self.logger.info(str(self.turn) + "still choose sample points")
                desire_distance = get_distance(curr_loc, safe_point)
                desire_angle = sympy.atan2(safe_point.y - curr_loc.y, safe_point.x - curr_loc.x)
        else:
            desire_distance = get_distance(curr_loc, safe_point)
            desire_angle = sympy.atan2(safe_point.y - curr_loc.y, safe_point.x - curr_loc.x)
        return (desire_distance, desire_angle)

    def go_for_middle_points_in_circle(self, prev_admissible, curr_loc, distance, angle, golf_map, quadrant, desire_angle, roll_factor):
        # 2. if we cannot use greedy, we try to find the points intersected with the golf map
        quadranty, quadrantx = quadrant
        if prev_admissible is None or prev_admissible or not self.remember_middle_points:
            circle = sympy.Circle(curr_loc, distance)
            # TODO: cost about 6-8 seconds, too slow
            intersect_points_origin = circle.intersection(golf_map)

            intersect_points_num = len(intersect_points_origin)
            temp_middle_points = []

            for i in range(intersect_points_num):
                for j in range(i + 1, intersect_points_num):
                    middle_point = shapely.geometry.Point(float(intersect_points_origin[i].x + intersect_points_origin[j].x) / 2,
                                                 float(intersect_points_origin[i].y + intersect_points_origin[j].y) / 2)
                    # find points that in the golf map polygon
                    if self.shapely_golf_map.contains(middle_point):
                        temp_middle_points.append(middle_point)

            if len(temp_middle_points) == 0:
                self.logger.error(str(self.turn) + "cannot find any middle point, BUG!!!")
                return (distance, angle, None)

            # if there are many ways to go, delete the points that can go back
            middle_points = []
            for i, middle_point in enumerate(temp_middle_points):
                curr_to_mid_quadranty, curr_to_mid_quadrantx = middle_point.y - curr_loc.y, middle_point.x - curr_loc.x
                if curr_to_mid_quadranty * quadranty < 0 or curr_to_mid_quadrantx * quadrantx < 0:
                    continue
                middle_points.append(middle_point)

            # if there we delete every point in temp_middle_points,
            # which means we could go longer ways than expected, we need to add back those points
            if len(middle_points) == 0:
                middle_points = temp_middle_points

            self.remember_middle_points = middle_points
        else:
            middle_points = list(self.remember_middle_points)

        middle_points_num = len(middle_points)
        curr2mid_angle = [0] * middle_points_num
        for i, middle_point in enumerate(middle_points):
            curr_to_mid_angle = sympy.atan2(middle_point.y - curr_loc.y, middle_point.x - curr_loc.x)
            curr2mid_angle[i] = abs(curr_to_mid_angle - desire_angle)

        distance_sorted_indexes = sorted(range(middle_points_num), key=lambda x: curr2mid_angle[x])

        middle_failed_times = [0] * middle_points_num

        midd_index = -1
        for i in distance_sorted_indexes:
            middle_point = middle_points[i]
            angle = sympy.atan2(middle_point.y - curr_loc.y, middle_point.x - curr_loc.x)
            for _ in range(self.simulate_times):
                is_succ, final_point = self.simulate_shapely_once(distance, angle, curr_loc, self.shapely_golf_map)
                if not is_succ:
                    middle_failed_times[i] += 1
                    if middle_failed_times[i] > self.tolerant_times:
                        middle_failed_times[i] = -1
                        break

            if middle_failed_times[i] != -1:
                midd_index = i
                break

        desire_distance = distance
        if midd_index != -1:
            self.logger.info(str(self.turn) + "select largest distance to middle point to go")
            desire_angle = sympy.atan2(middle_points[midd_index].y - curr_loc.y,
                                       middle_points[midd_index].x - curr_loc.x)

            return (desire_distance, desire_angle, self.simulate_times)

        # 3. if middle points are still not safe, choose the closest one to the target
        closest_index = distance_sorted_indexes[0]
        closest_middle_point = middle_points[closest_index]
        curr_to_mid = get_distance(closest_middle_point, curr_loc)
        desire_distance = None
        curr_to_mid = float(sympy.Min(constants.max_dist + self.skill, curr_to_mid / roll_factor))
        desire_angle = sympy.atan2(closest_middle_point.y - curr_loc.y, closest_middle_point.x - curr_loc.x)

        # try to step back until reached a safe state
        delta_times = 100
        delta = curr_to_mid / delta_times
        unsafe_distance2score = dict()
        count = 0
        while True:
            delte_distance = curr_to_mid - delta * count
            if count > delta_times / 2:
                break
            succ_times = 0
            for _ in range(self.simulate_times):
                is_succ, _ = self.simulate_shapely_once(delte_distance, desire_angle, curr_loc,self.shapely_golf_map)
                succ_times += is_succ

            if succ_times / self.simulate_times >= 1 - self.risk:
                desire_distance = delte_distance
                break

            unsafe_distance2score[delte_distance] = succ_times
            count += 1

        if desire_distance is None:
            self.logger.info(str(self.turn) + "risky!!! select closest middle point to go")
            unsafe_points = sorted(unsafe_distance2score.items(), key=lambda x: -x[1])
            desire_distance = unsafe_points[0][0]
            max_succ_times = unsafe_points[0][1]
        else:
            self.logger.info(str(self.turn) + "select closest middle point to go")
            max_succ_times = self.simulate_times

        return (desire_distance, desire_angle, max_succ_times)