import numpy as np
import sympy
from shapely.geometry import Polygon, Point
import skgeom as sg
from skgeom.draw import draw
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
import math
import logging
from typing import Tuple
from collections import defaultdict
import time

def draw_skeleton(polygon, skeleton, show_time=False):
    draw(polygon)

    for h in skeleton.halfedges:
        if h.is_bisector:
            p1 = h.vertex.point
            p2 = h.opposite.vertex.point

            """ m = (p1.y() - p2.y()) / (p1.x() - p2.x())
            b = p1.y() - (p1.x() * m) """
            # y = mx + b
            
            #plt.plot([p1.x(), p2.x()], [p1.y(), p2.y()], 'r-', lw=2)
            #plt.plot(p1.x(), p1.y(), 'bo')
            #plt.plot(p2.x(), p2.y(), 'bo')
    
    for v in skeleton.vertices:
        if (v.point not in polygon.vertices):
            plt.plot(v.point.x(), v.point.y(), 'bo')
    plt.savefig('test.png')

    if show_time:
        for v in skeleton.vertices:
            plt.gcf().gca().add_artist(plt.Circle(
                (v.point.x(), v.point.y()),
                v.time, color='blue', fill=False))

class Player:
    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger) -> None:
        """Initialise the player with given skill.

        Args:
            skill (int): skill of your player
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
            logger (logging.Logger): logger use this like logger.info("message")
        """
        self.skill = skill
        self.rng = rng
        self.logger = logger
        self.straight_skel_pts = []

    def play(self, score: int, golf_map: sympy.Polygon, target: sympy.geometry.Point2D, curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D, prev_landing_point: sympy.geometry.Point2D, prev_admissible: bool) -> Tuple[float, float]:
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
        required_dist = curr_loc.distance(target)

        since = time.time()
        #self.shapely_poly = Polygon([(p.x, p.y) for p in golf_map.vertices])
        poly = sg.Polygon([(p.x, p.y) for p in golf_map.vertices])
        #skel = sg.skeleton.create_interior_straight_skeleton(poly)
        skel = sg.skeleton.create_exterior_straight_skeleton(poly, 0.1)

        draw_skeleton(poly,skel)

        for h in skel.halfedges:
            if h.is_bisector:
                p1 = h.vertex.point
                #print(p1)
                p2 = h.opposite.vertex.point
                self.straight_skel_pts.append((p1.x(), p1.y()))
        
        print("time for construct_nodes:", time.time() - since)
        #image = np.array([self.straight_skel_pts], dtype=np.uint8)
        #imsave("test.png", image)


        #poly = sg.Polygon([sg.Point2(0, 0), sg.Point2(0, 3), sg.Point2(3, 3)])

        roll_factor = 1.1
        if required_dist < 20:
            roll_factor  = 1.0

        
        distance = sympy.Min(200+self.skill, required_dist/roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        return (distance, angle)
