#!/usr/bin/env python3
"""
waypoint_flight_feet_teamlogic.py

Standalone Crazyflie waypoint follower for a 5 ft x 5 ft clickable map.

This keeps the same user-facing interface as the previous clickable-map version,
but the flight control section is intentionally reorganized to look and behave
more like the team's ROS 2 waypoint code:

- A UnicastGoTo-style flight class
- TAKEOFF_Z / GOTO_TIME / GOTO_WAIT style constants
- wait_for_initial_position()
- goto_req(...)
- run()
- relative waypoint list converted into absolute go_to targets using:
      init_x + rx, init_y + ry, init_z + rz

The ROS service calls are replaced with cflib HighLevelCommander calls because
this file is meant to run directly from Python/Crazyradio without ROS 2.

Install:
    pip install cflib

Run:
    python waypoint_flight_feet_teamlogic.py

Optional URI:
    python waypoint_flight_feet_teamlogic.py --uri radio://0/80/2M/E7E7E7E7E7
"""

from __future__ import annotations

import argparse
import builtins
import csv
import math
import os
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Iterable, List, Optional, Tuple

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

# Reuse the same sensor/CSV/live-telemetry logic as lawnmower_flight.py
# so waypoint missions produce the same data files and live front-end telemetry.
from lawnmower_flight import (
    SensorLogger,
    StateEstimate,
    parse_temp_user,
    start_state_logging,
    configure_cf,
    reset_estimator,
    WAIT_FOR_ESTIMATE_TIMEOUT_S,
)


FT_TO_M = 0.3048
MAP_SIZE_FT = 5.0
HALF_MAP_FT = MAP_SIZE_FT / 2.0
DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"
DEFAULT_MAP_YAML = str(Path(__file__).resolve().parent / "maps" / "greenhouse_map.yaml")


@dataclass(frozen=True)
class Waypoint:
    """A waypoint on the 5x5 ft map, relative to the drone's start position."""

    x_ft: float
    y_ft: float
    z_ft: float

    @property
    def x_m(self) -> float:
        return self.x_ft * FT_TO_M

    @property
    def y_m(self) -> float:
        return self.y_ft * FT_TO_M

    @property
    def z_m(self) -> float:
        return self.z_ft * FT_TO_M


@dataclass(frozen=True)
class Position:
    x: float
    y: float
    z: float


class PositionWatcher:
    """Continuously watches stateEstimate.x/y/z from the Crazyflie log system."""

    def __init__(self, cf: Crazyflie):
        self._cf = cf
        self._position: Optional[Position] = None
        self._event = threading.Event()
        self._logconf = LogConfig(name="Position", period_in_ms=100)
        self._logconf.add_variable("stateEstimate.x", "float")
        self._logconf.add_variable("stateEstimate.y", "float")
        self._logconf.add_variable("stateEstimate.z", "float")

    @property
    def position(self) -> Optional[Position]:
        return self._position

    def _log_data(self, _timestamp, data, _logconf):
        self._position = Position(
            x=float(data["stateEstimate.x"]),
            y=float(data["stateEstimate.y"]),
            z=float(data["stateEstimate.z"]),
        )
        self._event.set()

    def _log_error(self, _logconf, msg):
        raise RuntimeError(f"Position log error: {msg}")

    def start(self):
        self._cf.log.add_config(self._logconf)
        self._logconf.data_received_cb.add_callback(self._log_data)
        self._logconf.error_cb.add_callback(self._log_error)
        self._logconf.start()

    def stop(self):
        try:
            self._logconf.stop()
        except Exception:
            pass

    def wait_for_position(self, timeout_sec: float = 10.0) -> Position:
        if not self._event.wait(timeout=timeout_sec) or self._position is None:
            raise RuntimeError(
                "No position estimate received. Make sure Lighthouse/positioning is working "
                "and stateEstimate.x/y/z are available."
            )
        return self._position


class StaticOccupancyMap:
    """
    Standalone version of the teammate's ROS OccupancyGrid path-safety logic.

    This intentionally keeps the same method names and behavior from
    goal_follower_node.py where possible:
      - world_to_map_cell(x, y)
      - cell_is_blocked(occupancy_value)
      - path_is_clear(goal_x, goal_y)
      - bresenham_line(x0, y0, x1, y1)

    The ROS /map subscription is replaced by loading the same map YAML + PGM
    files directly from disk.
    """

    def __init__(
        self,
        yaml_path: str,
        occupancy_block_threshold: int = 65,
        treat_unknown_as_obstacle: bool = True,
        logger=print,
    ):
        self.yaml_path = str(yaml_path)
        self.occupancy_block_threshold = int(occupancy_block_threshold)
        self.treat_unknown_as_obstacle = bool(treat_unknown_as_obstacle)
        self.log = logger

        self.resolution = 0.05
        self.origin = (-2.3, -2.301672, 0.0)
        self.width = 0
        self.height = 0
        self.data: List[int] = []
        self.current_position: Optional[Tuple[float, float, float]] = None

        self._load_map_from_yaml(self.yaml_path)

    @staticmethod
    def _parse_simple_yaml(path: str) -> dict:
        """Small YAML parser for ROS map YAML files so PyYAML is not required."""
        result = {}
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    result[key] = [float(v.strip()) for v in value[1:-1].split(",") if v.strip()]
                else:
                    try:
                        result[key] = float(value) if "." in value else int(value)
                    except ValueError:
                        result[key] = value.strip('"').strip("'")
        return result

    @staticmethod
    def _read_pgm(path: str) -> Tuple[int, int, List[int]]:
        """Read ROS map PGM images in either ASCII P2 or binary P5 format."""
        def next_token(f):
            token = b""
            while True:
                ch = f.read(1)
                if not ch:
                    return token.decode("ascii") if token else ""
                if ch == b"#":
                    f.readline()
                    continue
                if ch.isspace():
                    if token:
                        return token.decode("ascii")
                    continue
                token += ch

        with open(path, "rb") as f:
            magic = next_token(f)
            if magic not in ("P2", "P5"):
                raise ValueError(f"Unsupported map image format {magic!r}; expected P2 or P5 PGM")
            width = int(next_token(f))
            height = int(next_token(f))
            maxval = int(next_token(f))
            if maxval <= 0:
                raise ValueError("Invalid PGM max value")

            if magic == "P5":
                if maxval > 255:
                    raise ValueError("Only 8-bit binary P5 PGM maps are supported")
                pixels = list(f.read(width * height))
                if len(pixels) != width * height:
                    raise ValueError("PGM file ended before all pixels were read")
                return width, height, pixels

            pixels = []
            for _ in range(width * height):
                tok = next_token(f)
                if tok == "":
                    raise ValueError("PGM file ended before all pixels were read")
                val = int(tok)
                if maxval != 255:
                    val = round((val / maxval) * 255)
                pixels.append(max(0, min(255, val)))
            return width, height, pixels

    def _load_map_from_yaml(self, yaml_path: str):
        cfg = self._parse_simple_yaml(yaml_path)
        map_image = cfg.get("image")
        if not map_image:
            raise ValueError(f"Map YAML is missing image: {yaml_path}")
        image_path = map_image if os.path.isabs(str(map_image)) else str(Path(yaml_path).resolve().parent / str(map_image))

        self.resolution = float(cfg.get("resolution", self.resolution))
        origin = cfg.get("origin", list(self.origin))
        self.origin = (float(origin[0]), float(origin[1]), float(origin[2]) if len(origin) > 2 else 0.0)
        occupied_thresh = float(cfg.get("occupied_thresh", 0.65))
        free_thresh = float(cfg.get("free_thresh", 0.196))
        negate = int(cfg.get("negate", 0))

        self.width, self.height, pixels = self._read_pgm(image_path)

        data: List[int] = []
        for pix in pixels:
            occ_prob = (pix / 255.0) if negate else ((255 - pix) / 255.0)
            if occ_prob > occupied_thresh:
                data.append(100)
            elif occ_prob < free_thresh:
                data.append(0)
            else:
                data.append(-1)
        self.data = data
        self.log(
            f"Loaded static map: {Path(yaml_path).name} "
            f"({self.width}x{self.height}, {self.resolution:.3f} m/cell, "
            f"origin=({self.origin[0]:.3f},{self.origin[1]:.3f}))"
        )

    @staticmethod
    def bresenham_line(x0: int, y0: int, x1: int, y1: int):
        """! @brief Returns a list of (x, y) coordinates from (x0, y0) to (x1, y1)."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0

        while True:
            yield x, y
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def world_to_map_cell(self, x: float, y: float):
        """! @brief Convert world coordinates into map cell indices."""
        mx = math.floor((x - float(self.origin[0])) / float(self.resolution))
        my = math.floor((y - float(self.origin[1])) / float(self.resolution))
        if mx < 0 or my < 0 or mx >= int(self.width) or my >= int(self.height):
            return None
        return mx, my

    def cell_is_blocked(self, occupancy_value: int) -> bool:
        """! @brief Check if a map cell is blocked."""
        if occupancy_value < 0:
            return bool(self.treat_unknown_as_obstacle)
        return occupancy_value >= int(self.occupancy_block_threshold)

    def path_is_clear(self, goal_x: float, goal_y: float) -> bool:
        if self.current_position is None:
            self.log("Ignoring goal: no current position available")
            return False

        start = self.world_to_map_cell(self.current_position[0], self.current_position[1])
        goal = self.world_to_map_cell(goal_x, goal_y)
        if start is None or goal is None:
            self.log("Ignoring goal: start or goal is outside static map bounds")
            return False

        width = int(self.width)
        for mx, my in self.bresenham_line(start[0], start[1], goal[0], goal[1]):
            occ = self.data[my * width + mx]
            if self.cell_is_blocked(occ):
                self.log(f"Ignoring goal: blocked map cell ({mx}, {my}) occupancy={occ}")
                return False
        return True


class CflibUnicastGoTo:
    """
    cflib version of the teammate's UnicastGoTo ROS node.

    Names, flow, and comments are kept close to the original where possible.
    The main difference is that ROS clients/services are replaced by direct
    HighLevelCommander calls.
    """

    TAKEOFF_Z = 0.5       # meters; overwritten from the GUI's feet input
    GOTO_TIME = 3.0
    GOTO_WAIT = 3.5

    def __init__(
        self,
        uri: str,
        square_rel: List[Tuple[float, float, float]],
        takeoff_z_m: float,
        goto_time: float,
        goto_wait: float,
        takeoff_duration: float,
        land_duration: float,
        logger=print,
        use_map_line_check: bool = False,
        map_yaml_path: str = DEFAULT_MAP_YAML,
        occupancy_block_threshold: int = 65,
        treat_unknown_as_obstacle: bool = True,
        ambient_room_temp: str = "72F",
    ):
        """!
        @brief Initialize connection parameters and odometry/state-estimate watcher.
        """
        self.uri = uri
        self.square_rel = square_rel
        self.TAKEOFF_Z = takeoff_z_m
        self.GOTO_TIME = goto_time
        self.GOTO_WAIT = goto_wait
        self.takeoff_duration = takeoff_duration
        self.land_duration = land_duration
        self.log = logger
        self.use_map_line_check = bool(use_map_line_check)
        self.map_yaml_path = map_yaml_path
        self.occupancy_block_threshold = int(occupancy_block_threshold)
        self.treat_unknown_as_obstacle = bool(treat_unknown_as_obstacle)
        self.ambient_room_temp = str(ambient_room_temp or "72F").strip()
        self.map_safety: Optional[StaticOccupancyMap] = None

        # Odom/state estimate
        self.initial_position = None
        self._cf: Optional[Crazyflie] = None
        self._watcher: Optional[PositionWatcher] = None

    def wait_for_services(self, timeout_sec: float = 10.0):
        """!
        @brief ROS version waited for takeoff, land, and go_to services.
        @details cflib has no ROS services, so opening SyncCrazyflie is the equivalent readiness check.
        """
        _ = timeout_sec
        return True

    def odom_cb(self, position: Position):
        """!
        @brief Odometry callback that captures the first pose as initial position.
        @param position Incoming position estimate.
        """
        if self.initial_position is None:
            self.initial_position = (float(position.x), float(position.y), float(position.z))
            self.log(
                f"Initial position captured: ({position.x:.3f}, {position.y:.3f}, {position.z:.3f})"
            )

    def wait_for_initial_position(self, timeout_sec: float = 10.0):
        """!
        @brief Block until initial position is available from odometry.
        @param timeout_sec Maximum wait time in seconds.
        @throws RuntimeError If no odometry is received before timeout and lands the drone.
        """
        if self._watcher is None:
            raise RuntimeError("Position watcher is not started")

        try:
            position = self._watcher.wait_for_position(timeout_sec=timeout_sec)
            self.odom_cb(position)
        except Exception:
            if self._cf is not None:
                try:
                    self._cf.high_level_commander.land(0.00, 1.0)
                    time.sleep(1.2)
                    self._cf.high_level_commander.stop()
                except Exception:
                    pass
            raise RuntimeError(f"No odometry received on stateEstimate.x/y/z within {timeout_sec}s")

    def call(self, label: str, action):
        """!
        @brief Compatibility wrapper matching the original call(client, request, label) idea.
        @throws RuntimeError If the command call fails.
        """
        try:
            return action()
        except Exception as exc:
            raise RuntimeError(f"{label} failed: {exc}") from exc

    def goto_req(self, x: float, y: float, z: float, yaw: float = 0.0, duration: Optional[float] = None):
        """!
        @brief Send an absolute go_to request and wait for completion window.
        @param x Goal x coordinate in meters.
        @param y Goal y coordinate in meters.
        @param z Goal z coordinate in meters.
        @param yaw Goal yaw in degrees.
        @param duration Requested trajectory duration in seconds.
        """
        if self._cf is None:
            raise RuntimeError("Crazyflie is not connected")
        duration = self.GOTO_TIME if duration is None else float(duration)

        self.call(
            "go_to",
            lambda: self._cf.high_level_commander.go_to(
                float(x),
                float(y),
                float(z),
                float(yaw),
                duration,
                relative=False,
            ),
        )
        time.sleep(max(duration, self.GOTO_WAIT))

    def run(self):
        """!
        @brief Execute takeoff, waypoint trajectory, and landing sequence.
        @details Waypoints are computed relative to initial odometry pose.
        """
        z = self.TAKEOFF_Z

        cflib.crtp.init_drivers()
        self.log(f"Connecting to {self.uri} ...")

        cf = Crazyflie(rw_cache="./cache")
        with SyncCrazyflie(self.uri, cf=cf) as scf:
            self._cf = scf.cf
            configure_cf(self._cf)

            # Use the same Lighthouse/state-estimate logging object as lawnmower_flight.py.
            # This gives both the mission code and SensorLogger the same live pose values.
            state = StateEstimate()
            pos_lg = start_state_logging(self._cf, state)
            sensor_logger = None

            try:
                self.wait_for_services()

                if not state.wait_for_first(WAIT_FOR_ESTIMATE_TIMEOUT_S):
                    raise RuntimeError("No stateEstimate received. Check Lighthouse/Kalman.")

                self.log("Keep the drone still. Resetting estimator...")
                reset_estimator(self._cf)
                # Wait for fresh post-reset samples before capturing the origin.
                # This prevents repeated flights from reusing stale final pose data.
                time.sleep(2.0)

                origin = state.pose
                self.initial_position = (float(origin.x), float(origin.y), float(origin.z))
                init_x, init_y, init_z = self.initial_position
                self.log(
                    f"Initial position captured: ({init_x:.3f}, {init_y:.3f}, {init_z:.3f})"
                )

                if self.use_map_line_check:
                    self.map_safety = StaticOccupancyMap(
                        self.map_yaml_path,
                        occupancy_block_threshold=self.occupancy_block_threshold,
                        treat_unknown_as_obstacle=self.treat_unknown_as_obstacle,
                        logger=self.log,
                    )
                    self.map_safety.current_position = (init_x, init_y, init_z)
                    self.log(
                        "Static obstacle / operational-area checks enabled "
                        f"(threshold={self.occupancy_block_threshold}, "
                        f"unknown_is_obstacle={self.treat_unknown_as_obstacle})"
                    )

                # Start the same CSV + live telemetry sensor logger used by lawnmower_flight.py.
                # It records before takeoff, during flight, and through landing.
                #
                # lawnmower_flight.SensorLogger.start() normally prompts with input().
                # That can block this GUI/front-end-launched script after the first run,
                # so we temporarily provide the GUI ambient value while keeping the
                # original SensorLogger code intact.
                sensor_logger = SensorLogger(scf, state)
                original_input = builtins.input
                try:
                    builtins.input = lambda prompt='': (self.log(f"{prompt}{self.ambient_room_temp}"), self.ambient_room_temp)[1]
                    sensor_logger.start()
                finally:
                    builtins.input = original_input

                self.log(f"Taking off to {z:.3f} m / {z / FT_TO_M:.2f} ft ...")
                self.call("takeoff", lambda: self._cf.high_level_commander.takeoff(float(z), self.takeoff_duration))
                time.sleep(self.takeoff_duration + 0.5)

                for idx, (rx, ry, rz) in enumerate(self.square_rel, start=1):
                    target_x = init_x + rx
                    target_y = init_y + ry
                    target_z = init_z + rz
                    self.log(
                        f"go_to {idx}/{len(self.square_rel)}: "
                        f"x={target_x:.3f}, y={target_y:.3f}, z={target_z:.3f}"
                    )

                    if self.use_map_line_check and self.map_safety is not None:
                        current_pose = state.pose
                        self.map_safety.current_position = (float(current_pose.x), float(current_pose.y), float(current_pose.z))
                        if not self.map_safety.path_is_clear(target_x, target_y):
                            self.log(
                                f"Waypoint {idx}/{len(self.square_rel)} rejected: "
                                "path crosses an obstacle or leaves the operational area"
                            )
                            continue

                    self.goto_req(target_x, target_y, target_z)

                self.log("Landing ...")
                self.call("land", lambda: self._cf.high_level_commander.land(0.00, self.land_duration))
                time.sleep(self.land_duration + 0.5)
                self._cf.high_level_commander.stop()
                self.log("Mission complete.")

            except Exception:
                try:
                    self.log("Exception occurred. Attempting emergency land ...")
                    self._cf.high_level_commander.land(0.00, self.land_duration)
                    time.sleep(self.land_duration + 0.5)
                    self._cf.high_level_commander.stop()
                finally:
                    raise
            finally:
                if sensor_logger is not None:
                    try:
                        sensor_logger.stop()
                    except Exception:
                        pass
                    time.sleep(0.5)
                try:
                    pos_lg.stop()
                except Exception:
                    pass
                self.map_safety = None
                self._cf = None

        self.log("Crazyflie link closed. Ready for another mission.")
        time.sleep(1.0)


class ClickWaypointPlanner(tk.Tk):
    """Tkinter interface for clicking waypoints on a 5 ft x 5 ft map."""

    def __init__(self, uri: str):
        super().__init__()
        self.title("Crazyflie 5x5 ft Click Waypoint Planner")
        self.geometry("1040x780")
        self.minsize(900, 650)

        self.uri = uri
        self.waypoints: List[Waypoint] = []
        self.status_queue: "queue.Queue[str]" = queue.Queue()
        self.flight_thread: Optional[threading.Thread] = None

        self.canvas_size_px = 560
        self.margin_px = 30
        self.map_px = self.canvas_size_px - 2 * self.margin_px

        # Everything visible to the user is now in feet.
        self.flight_height_ft_var = tk.DoubleVar(value=1.64)  # about 0.50 m
        self.ambient_room_temp_var = tk.StringVar(value="72F")
        self.goto_duration_var = tk.DoubleVar(value=3.0)
        self.goto_wait_var = tk.DoubleVar(value=3.5)
        # Takeoff/landing durations are computed automatically from flight height.
        # Minimum is 2 seconds; higher altitudes get slower vertical motion.
        self.return_home_var = tk.BooleanVar(value=True)
        self.dry_run_var = tk.BooleanVar(value=True)
        self.use_map_line_check_var = tk.BooleanVar(value=True)
        self.map_yaml_path_var = tk.StringVar(value=DEFAULT_MAP_YAML)
        self.occupancy_block_threshold_var = tk.IntVar(value=65)
        self.treat_unknown_as_obstacle_var = tk.BooleanVar(value=True)
        self.uri_var = tk.StringVar(value=self.uri)

        self._build_ui()
        self._draw_map()
        self.after(100, self._poll_status_queue)

    def _build_ui(self):
        outer = tk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)

        scroll_canvas = tk.Canvas(outer, highlightthickness=0)
        v_scroll = tk.Scrollbar(outer, orient=tk.VERTICAL, command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=v_scroll.set)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        root = tk.Frame(scroll_canvas, padx=12, pady=12)
        root_window = scroll_canvas.create_window((0, 0), window=root, anchor="nw")

        def _update_scroll_region(_event=None):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
            scroll_canvas.itemconfigure(root_window, width=scroll_canvas.winfo_width())

        root.bind("<Configure>", _update_scroll_region)
        scroll_canvas.bind("<Configure>", _update_scroll_region)
        scroll_canvas.bind_all("<MouseWheel>", lambda e: scroll_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        left = tk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        self.canvas = tk.Canvas(left, width=self.canvas_size_px, height=self.canvas_size_px, bg="white")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        help_text = (
            "Click to add waypoints. Center = drone start position. "
            "Map is 5 ft x 5 ft, from -2.5 to +2.5 ft on each axis."
        )
        tk.Label(left, text=help_text, wraplength=self.canvas_size_px, justify=tk.LEFT).pack(pady=(8, 0))

        right = tk.Frame(root, padx=18)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(right, text="Flight Settings", font=("Arial", 14, "bold")).pack(anchor="w")

        self._labeled_entry(right, "Crazyflie URI", self.uri_var)
        self._labeled_entry(right, "Flight height Z (feet)", self.flight_height_ft_var)
        self._labeled_entry(right, "Ambient room temp for sensor logging (e.g. 72F or 22C)", self.ambient_room_temp_var)
        self._labeled_entry(right, "go_to duration per waypoint (sec)", self.goto_duration_var)
        self._labeled_entry(right, "go_to wait window (sec)", self.goto_wait_var)
        tk.Label(
            right,
            text="Takeoff/landing duration: automatic from height (min 2 sec)",
            fg="#555555",
        ).pack(anchor="w", pady=(8, 0))

        tk.Checkbutton(
            right,
            text="Return to start before landing",
            variable=self.return_home_var,
            command=self._refresh_waypoints,
        ).pack(anchor="w", pady=(8, 0))

        tk.Checkbutton(
            right,
            text="Dry run only: print converted waypoints but do not connect/fly",
            variable=self.dry_run_var,
        ).pack(anchor="w", pady=(4, 6))

        tk.Checkbutton(
            right,
            text="Use static map obstacle / operational-area check",
            variable=self.use_map_line_check_var,
        ).pack(anchor="w", pady=(2, 0))
        self._labeled_entry(right, "Static map YAML", self.map_yaml_path_var)
        self._labeled_entry(right, "Occupancy block threshold", self.occupancy_block_threshold_var)
        tk.Checkbutton(
            right,
            text="Treat unknown map cells as obstacles",
            variable=self.treat_unknown_as_obstacle_var,
        ).pack(anchor="w", pady=(4, 10))

        button_frame = tk.Frame(right)
        button_frame.pack(anchor="w", pady=(4, 10))
        tk.Button(button_frame, text="Undo Point", command=self.undo_point).grid(row=0, column=0, padx=(0, 6), pady=3)
        tk.Button(button_frame, text="Clear", command=self.clear_points).grid(row=0, column=1, padx=(0, 6), pady=3)
        tk.Button(button_frame, text="Save CSV", command=self.save_csv).grid(row=0, column=2, padx=(0, 6), pady=3)
        tk.Button(button_frame, text="Load CSV", command=self.load_csv).grid(row=0, column=3, padx=(0, 6), pady=3)

        tk.Button(
            right,
            text="FLY WAYPOINTS",
            command=self.start_flight,
            bg="#ffdddd",
            font=("Arial", 12, "bold"),
        ).pack(anchor="w", fill=tk.X, pady=(4, 12))

        tk.Label(right, text="Waypoints", font=("Arial", 12, "bold")).pack(anchor="w")
        self.waypoint_list = tk.Listbox(right, height=12)
        self.waypoint_list.pack(fill=tk.X, pady=(4, 12))

        tk.Label(right, text="Status", font=("Arial", 12, "bold")).pack(anchor="w")
        status_frame = tk.Frame(right)
        status_frame.pack(fill=tk.BOTH, expand=True)
        self.status = tk.Text(status_frame, height=14, wrap=tk.WORD)
        status_scroll = tk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status.yview)
        self.status.configure(yscrollcommand=status_scroll.set)
        self.status.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._log("Ready. Dry run is enabled by default for safety.")

    @staticmethod
    def _labeled_entry(parent: tk.Widget, label: str, variable):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=(8, 0))
        tk.Label(frame, text=label).pack(anchor="w")
        tk.Entry(frame, textvariable=variable).pack(fill=tk.X)

    def _draw_map(self):
        self.canvas.delete("all")
        m = self.margin_px
        end = self.margin_px + self.map_px

        self.canvas.create_rectangle(m, m, end, end, outline="black", width=2)

        for i in range(6):
            x = m + i * self.map_px / MAP_SIZE_FT
            y = m + i * self.map_px / MAP_SIZE_FT
            self.canvas.create_line(x, m, x, end, fill="#dddddd")
            self.canvas.create_line(m, y, end, y, fill="#dddddd")

        cx, cy = self.ft_to_canvas(0.0, 0.0)
        self.canvas.create_line(cx, m, cx, end, fill="#888888", width=2)
        self.canvas.create_line(m, cy, end, cy, fill="#888888", width=2)
        self.canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill="green", outline="green")
        self.canvas.create_text(cx + 42, cy - 12, text="START", fill="green", font=("Arial", 10, "bold"))

        self.canvas.create_text((m + end) / 2, 12, text="+Y / Forward")
        self.canvas.create_text((m + end) / 2, self.canvas_size_px - 12, text="-Y / Backward")
        self.canvas.create_text(17, (m + end) / 2, text="-X", angle=90)
        self.canvas.create_text(self.canvas_size_px - 17, (m + end) / 2, text="+X", angle=90)

        for ft in [-2, -1, 0, 1, 2]:
            x, _ = self.ft_to_canvas(float(ft), 0.0)
            _, y = self.ft_to_canvas(0.0, float(ft))
            self.canvas.create_text(x, end + 14, text=f"{ft} ft")
            self.canvas.create_text(m - 18, y, text=f"{ft}")

        self._draw_waypoints()

    def ft_to_canvas(self, x_ft: float, y_ft: float) -> Tuple[float, float]:
        m = self.margin_px
        x_px = m + ((x_ft + HALF_MAP_FT) / MAP_SIZE_FT) * self.map_px
        y_px = m + ((HALF_MAP_FT - y_ft) / MAP_SIZE_FT) * self.map_px
        return x_px, y_px

    def canvas_to_ft(self, x_px: float, y_px: float) -> Tuple[float, float]:
        m = self.margin_px
        x_ft = ((x_px - m) / self.map_px) * MAP_SIZE_FT - HALF_MAP_FT
        y_ft = HALF_MAP_FT - ((y_px - m) / self.map_px) * MAP_SIZE_FT
        x_ft = max(-HALF_MAP_FT, min(HALF_MAP_FT, x_ft))
        y_ft = max(-HALF_MAP_FT, min(HALF_MAP_FT, y_ft))
        return x_ft, y_ft

    def _on_canvas_click(self, event):
        x_ft, y_ft = self.canvas_to_ft(event.x, event.y)
        z_ft = float(self.flight_height_ft_var.get())
        self.waypoints.append(Waypoint(x_ft=x_ft, y_ft=y_ft, z_ft=z_ft))
        self._refresh_waypoints()

    def _draw_waypoints(self):
        if not self.waypoints:
            return
        previous = (0.0, 0.0)
        for idx, waypoint in enumerate(self.waypoints, start=1):
            x_px, y_px = self.ft_to_canvas(waypoint.x_ft, waypoint.y_ft)
            prev_px = self.ft_to_canvas(*previous)
            self.canvas.create_line(prev_px[0], prev_px[1], x_px, y_px, fill="blue", width=2, arrow=tk.LAST)
            self.canvas.create_oval(x_px - 7, y_px - 7, x_px + 7, y_px + 7, fill="blue", outline="blue")
            self.canvas.create_text(x_px + 12, y_px - 12, text=str(idx), fill="blue", font=("Arial", 10, "bold"))
            previous = (waypoint.x_ft, waypoint.y_ft)

        if self.return_home_var.get():
            start_px = self.ft_to_canvas(0.0, 0.0)
            last_px = self.ft_to_canvas(self.waypoints[-1].x_ft, self.waypoints[-1].y_ft)
            self.canvas.create_line(last_px[0], last_px[1], start_px[0], start_px[1], fill="#55aa55", width=2, dash=(4, 4), arrow=tk.LAST)

    def _refresh_waypoints(self):
        self._draw_map()
        self.waypoint_list.delete(0, tk.END)
        for idx, wp in enumerate(self.waypoints, start=1):
            self.waypoint_list.insert(
                tk.END,
                f"{idx:02d}: x={wp.x_ft:+.2f} ft ({wp.x_m:+.3f} m), "
                f"y={wp.y_ft:+.2f} ft ({wp.y_m:+.3f} m), "
                f"z={wp.z_ft:.2f} ft ({wp.z_m:.3f} m)",
            )

    def undo_point(self):
        if self.waypoints:
            self.waypoints.pop()
            self._refresh_waypoints()

    def clear_points(self):
        self.waypoints.clear()
        self._refresh_waypoints()

    def save_csv(self):
        if not self.waypoints:
            messagebox.showinfo("No waypoints", "Add at least one waypoint first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_ft", "y_ft", "z_ft"])
            for wp in self.waypoints:
                writer.writerow([wp.x_ft, wp.y_ft, wp.z_ft])
        self._log(f"Saved {len(self.waypoints)} waypoint(s) to {path}")

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        loaded: List[Waypoint] = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "z_ft" in row and row["z_ft"]:
                    z_ft = float(row["z_ft"])
                elif "z_m" in row and row["z_m"]:
                    z_ft = float(row["z_m"]) / FT_TO_M
                else:
                    z_ft = float(self.flight_height_ft_var.get())
                loaded.append(Waypoint(x_ft=float(row["x_ft"]), y_ft=float(row["y_ft"]), z_ft=z_ft))
        self.waypoints = loaded
        self._refresh_waypoints()
        self._log(f"Loaded {len(self.waypoints)} waypoint(s) from {path}")


    @staticmethod
    def _auto_takeoff_duration(height_ft: float) -> float:
        """Compute a safe takeoff duration from height in feet. Minimum 2 seconds."""
        return max(2.0, float(height_ft) / 1.0)

    @staticmethod
    def _auto_land_duration(height_ft: float) -> float:
        """Compute a safe landing duration from height in feet. Minimum 2 seconds.

        Landing is intentionally slower than takeoff because 1 second landings
        were causing hard impacts.
        """
        return max(2.0, float(height_ft) / 0.75)

    def start_flight(self):
        if self.flight_thread and self.flight_thread.is_alive():
            messagebox.showwarning("Flight running", "A flight is already running.")
            return
        if not self.waypoints:
            messagebox.showwarning("No waypoints", "Click at least one waypoint first.")
            return

        if not self.dry_run_var.get():
            ok = messagebox.askyesno(
                "Confirm real flight",
                "Dry run is OFF. The drone will take off and fly the selected path.\n\n"
                "Confirm that the area is clear, positioning is working, battery is charged, "
                "and you are ready to emergency stop if needed.",
            )
            if not ok:
                return

        self.uri = self.uri_var.get().strip() or DEFAULT_URI
        self.flight_thread = threading.Thread(target=self._run_flight_thread, daemon=True)
        self.flight_thread.start()

    def _run_flight_thread(self):
        try:
            settings = {
                "flight_height_ft": float(self.flight_height_ft_var.get()),
                "flight_height_m": float(self.flight_height_ft_var.get()) * FT_TO_M,
                "ambient_room_temp": self.ambient_room_temp_var.get().strip() or "72F",
                "goto_duration_sec": float(self.goto_duration_var.get()),
                "goto_wait_sec": float(self.goto_wait_var.get()),
                "takeoff_duration_sec": self._auto_takeoff_duration(float(self.flight_height_ft_var.get())),
                "land_duration_sec": self._auto_land_duration(float(self.flight_height_ft_var.get())),
                "return_home": bool(self.return_home_var.get()),
                "dry_run": bool(self.dry_run_var.get()),
                "use_map_line_check": bool(self.use_map_line_check_var.get()),
                "map_yaml_path": self.map_yaml_path_var.get().strip(),
                "occupancy_block_threshold": int(self.occupancy_block_threshold_var.get()),
                "treat_unknown_as_obstacle": bool(self.treat_unknown_as_obstacle_var.get()),
            }

            # Validate ambient input before the Crazyflie connection starts.
            parse_temp_user(settings["ambient_room_temp"])

            waypoints = [Waypoint(wp.x_ft, wp.y_ft, settings["flight_height_ft"]) for wp in self.waypoints]
            if settings["return_home"]:
                waypoints.append(Waypoint(0.0, 0.0, settings["flight_height_ft"]))

            self._run_waypoint_mission(self.uri, waypoints, settings)
        except Exception as exc:
            self._thread_log(f"ERROR: {exc}")
        finally:
            self.flight_thread = None

    def _run_waypoint_mission(self, uri: str, waypoints: List[Waypoint], settings: dict):
        """
        Execute the waypoint mission.

        This is now structured very close to the team's ROS test_node.py:
        - build a square_rel-style list of relative waypoints in meters
        - create a UnicastGoTo-style flight object
        - call run(), which does wait_for_initial_position(), takeoff, goto_req loop, and land
        """
        self._thread_log("Mission waypoints, relative to starting position:")
        self._thread_log(
            f"Auto vertical timing: takeoff={settings['takeoff_duration_sec']:.1f}s, "
            f"land={settings['land_duration_sec']:.1f}s"
        )
        self._thread_log(f"Ambient temp for sensor logger: {settings['ambient_room_temp']}")
        for i, wp in enumerate(waypoints, start=1):
            self._thread_log(
                f"  {i:02d}: x={wp.x_ft:+.2f} ft / {wp.x_m:+.3f} m, "
                f"y={wp.y_ft:+.2f} ft / {wp.y_m:+.3f} m, "
                f"z={wp.z_ft:.2f} ft / {wp.z_m:.3f} m"
            )

        # Same shape as teammate's square_rel list, but filled from the clicked points.
        square_rel = [(wp.x_m, wp.y_m, wp.z_m) for wp in waypoints]

        if settings["use_map_line_check"]:
            try:
                _ = StaticOccupancyMap(
                    settings["map_yaml_path"],
                    occupancy_block_threshold=settings["occupancy_block_threshold"],
                    treat_unknown_as_obstacle=settings["treat_unknown_as_obstacle"],
                    logger=self._thread_log,
                )
                self._thread_log("Static map loaded successfully. Full path checks run after the real start position is known.")
            except Exception as exc:
                self._thread_log(f"ERROR loading static map: {exc}")
                return

        if settings["dry_run"]:
            self._thread_log("Dry run complete. No Crazyflie connection was opened.")
            return

        node = CflibUnicastGoTo(
            uri=uri,
            square_rel=square_rel,
            takeoff_z_m=settings["flight_height_m"],
            goto_time=settings["goto_duration_sec"],
            goto_wait=settings["goto_wait_sec"],
            takeoff_duration=settings["takeoff_duration_sec"],
            land_duration=settings["land_duration_sec"],
            logger=self._thread_log,
            use_map_line_check=settings["use_map_line_check"],
            map_yaml_path=settings["map_yaml_path"],
            occupancy_block_threshold=settings["occupancy_block_threshold"],
            treat_unknown_as_obstacle=settings["treat_unknown_as_obstacle"],
            ambient_room_temp=settings["ambient_room_temp"],
        )
        node.run()

    def _thread_log(self, text: str):
        self.status_queue.put(text)

    def _log(self, text: str):
        self.status.insert(tk.END, text + "\n")
        self.status.see(tk.END)

    def _poll_status_queue(self):
        while True:
            try:
                self._log(self.status_queue.get_nowait())
            except queue.Empty:
                break
        self.after(100, self._poll_status_queue)


# Kept from teammate's ROS goal follower logic for future map obstacle checks.
def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> Iterable[Tuple[int, int]]:
    """! @brief Returns a list of (x, y) coordinates from (x0, y0) to (x1, y1)."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0

    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def parse_args():
    parser = argparse.ArgumentParser(description="Crazyflie clickable 5x5 ft waypoint follower")
    parser.add_argument(
        "--uri",
        default=uri_helper.uri_from_env(default=DEFAULT_URI),
        help=f"Crazyflie URI. Default: {DEFAULT_URI}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = ClickWaypointPlanner(uri=args.uri)
    app.mainloop()


if __name__ == "__main__":
    main()
