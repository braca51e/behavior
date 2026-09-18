"""Perception: odometry, room prior, predicate detectors (onboard-only at eval).

CPU-testable; the optional heavy VO model is imported lazily.  Nothing here
imports from :mod:`b1k.policy` or privileged/training code.
"""
from .odometry import Odometry, OccupancyGrid
from .rooms import RoomPrior

__all__ = ["Odometry", "OccupancyGrid", "RoomPrior"]
