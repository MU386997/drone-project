import time
import math
import csv
from dataclasses import dataclass
from datetime import datetime
from threading import Event
from typing import Optional

import cflib.crtp
from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger

# ============================================================
#                 FLIGHT (EASY-TO-MODIFY) PARAMETERS
# ============================================================
URI = uri_helper.uri_from_env(default="radio://0/80/2M")  # match your sensor script URI style

SQUARE_SIDE_FT = 4.0
STEP_FT = 0.5
ALT_LAYERS_FT = [1.5, 3.0]

TAKEOFF_TIME_S = 2.5
LAND_TIME_S = 3.0
SPEED_MPS = 0.35

DWELL_TIME_S = 1.0          # <-- requested: dwell at each waypoint
POSITION_TOL_M = 0.05       # arrival tolerance (meters)

LEFT_IS_NEG_X = True
BACK_IS_NEG_Y = True

MAX_Z_M = 1.6               # safety cap (~5.25 ft)
WAIT_FOR_ESTIMATE_TIMEOUT_S = 8.0
STATE_LOG_PERIOD_MS = 50    # for position arrival check

# ============================================================
#              SENSOR LOGGING PARAMETERS (UNCHANGED)
# ============================================================
P_STANDARD = 1013.25  # hPa
R_DRY_AIR = 287.05    # J/(kg*K)
STATION_ELEV_M = 15   # Alexandria, VA (for QFF reduction)

LOG_PERIOD_MS = 500   # 2 Hz (sensor logging rate)
CSV_FILE = "drone_mapping_data.csv"
CSV_FLUSH_EVERY = 20
PRINT_EVERY_S = 0.0   # print every reading at 2 Hz

# ============================================================
#                       Reovery / Geofencing
# ============================================================
from threading import Event, Thread, Lock
 # ============================================================
#                    GEOFENCE / FAILSAFE
# ============================================================

# 4x4 meter box centered on takeoff origin
GEOFENCE_X_SIZE_M = 4.0
GEOFENCE_Y_SIZE_M = 4.0

# Vertical limits 
GEOFENCE_Z_MIN_M = 0.0
GEOFENCE_Z_MAX_M = MAX_Z_M

# This is the variable to intermittently check for positioning. 
GEOFENCE_CHECK_PERIOD_S = 0.10

# Altitude when the drone is called to recover. 
RECOVERY_HOVER_Z_M = 0.6

# Added buffer to extend beyond the area the drone is scanning over. 
GEOFENCE_MARGIN_M = 0.1

# ============================================================
#                 GEOFENCE / RECOVERY HELPERS
# ============================================================

class MissionAbort(Exception):
    """Raised when the geofence monitor forces a recovery/abort."""
    pass


def build_geofence_from_origin(origin: Pose):
    """
    Create a 4x4m XY geofence centered on the takeoff origin.
    """
    half_x = GEOFENCE_X_SIZE_M / 2.0
    half_y = GEOFENCE_Y_SIZE_M / 2.0

    return {
        "x_min": origin.x - half_x,
        "x_max": origin.x + half_x,
        "y_min": origin.y - half_y,
        "y_max": origin.y + half_y,
        "z_min": GEOFENCE_Z_MIN_M,
        "z_max": GEOFENCE_Z_MAX_M,
    }


def pose_within_geofence(p: Pose, fence: dict, margin_m: float = GEOFENCE_MARGIN_M) -> bool:
    return (
        (fence["x_min"] - margin_m) <= p.x <= (fence["x_max"] + margin_m) and
        (fence["y_min"] - margin_m) <= p.y <= (fence["y_max"] + margin_m) and
        (fence["z_min"] - margin_m) <= p.z <= (fence["z_max"] + margin_m)
    )


def recover_to_origin(cf: Crazyflie, state: StateEstimate, origin: Pose):
    """
    Failsafe recovery:
    1) stop current motion
    2) go to takeoff XY at a safe hover altitude
    3) land
    """
    hl = cf.high_level_commander
    current = state.pose

    safe_z = min(max(RECOVERY_HOVER_Z_M, 0.25), MAX_Z_M)
    if current.z > safe_z:
        safe_z = min(current.z, MAX_Z_M)

    print("\n[FAILSAFE] Geofence violation detected.")
    print(f"[FAILSAFE] Current pose: x={current.x:.3f}, y={current.y:.3f}, z={current.z:.3f}")
    print(f"[FAILSAFE] Recovering to takeoff origin: x0={origin.x:.3f}, y0={origin.y:.3f}, z_hover={safe_z:.3f}")

    try:
        # Stop previous trajectory commands
        hl.stop()
        time.sleep(0.2)

        # Move above origin first
        target = Pose(origin.x, origin.y, safe_z)
        dur = compute_duration(current, target, SPEED_MPS)
        hl.go_to(target.x, target.y, target.z, yaw=0.0, duration_s=dur, relative=False)

        start_time = time.time()
        timeout = dur + 3.0
        while time.time() - start_time < timeout:
            p = state.pose
            d = math.sqrt((p.x - target.x) ** 2 + (p.y - target.y) ** 2 + (p.z - target.z) ** 2)
            if d < POSITION_TOL_M:
                break
            time.sleep(0.05)

    except Exception as e:
        print(f"[FAILSAFE] Recovery go_to had an issue: {e}")

    finally:
        print("[FAILSAFE] Landing at/near origin...")
        try:
            hl.land(0.0, LAND_TIME_S)
            time.sleep(LAND_TIME_S + 0.3)
        finally:
            try:
                hl.stop()
            except Exception:
                pass


def geofence_monitor_loop(
    cf: Crazyflie,
    state: StateEstimate,
    origin: Pose,
    fence: dict,
    stop_event: Event,
    recovery_started: Event,
):
    """
    Background monitor that checks state.pose every GEOFENCE_CHECK_PERIOD_S.
    If the drone leaves the geofence, it starts recovery once.
    """
    while not stop_event.is_set():
        p = state.pose

        if not pose_within_geofence(p, fence):
            if not recovery_started.is_set():
                recovery_started.set()
                recover_to_origin(cf, state, origin)
            return

        time.sleep(GEOFENCE_CHECK_PERIOD_S)

# ============================================================
#                       HELPERS
# ============================================================
def ft_to_m(ft: float) -> float:
    return ft * 0.3048

def m_to_ft(m: float) -> float:
    return m * 3.28084

def c_to_f(c: float) -> float:
    return (c * 9.0 / 5.0) + 32.0

def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0

def parse_temp_user(s: str) -> tuple[float, str]:
    """Parse ambient temperature like '74F' or '22C' (case-insensitive).
    Returns (temperature_in_celsius, user_unit 'C' or 'F').
    """
    if s is None:
        raise ValueError("empty")
    s = s.strip().upper().replace(" ", "")
    if not s:
        raise ValueError("empty")

    # Allow raw number with no suffix -> assume C
    if s[-1] not in ("C", "F"):
        return float(s), "C"

    unit = s[-1]
    val = float(s[:-1])
    return (f_to_c(val), "F") if unit == "F" else (val, "C")

# ============================================================
#         SENSOR LOGIC
# ============================================================
class ThermalEngine:
    """Thermal correction engine with two-timescale tracking."""
    def __init__(self, initial_raw_c: float, initial_ambient_c: float):
        # Filtering
        self.alpha = 0.12
        self.ema_temp_c = initial_raw_c
        self.prev_ema_temp_c = initial_raw_c

        # Two-timescale estimates
        self.ambient_est_c = initial_ambient_c
        self.idle_offset_c = initial_raw_c - initial_ambient_c

        # Mode/state machine
        self.mode = "STABLE"  # STABLE | TRANSITION
        self.state = "IDLE"
        self.last_event = "Initialized"
        self.last_update_time = time.time()

        # Thresholds / tunables (idle-focused)
        self.heat_thresh = 0.07
        self.cool_thresh = 0.03
        self.roc_stable_thresh = 0.02
        self.drift_band_c = 0.45
        self.env_error_c = 1.00
        self.ambient_cool_margin_c = 0.30

        # Gains
        self.offset_gain = 0.05
        self.ambient_gain_stable = 0.03
        self.ambient_gain_transition = 0.15

        # Counters for hysteresis
        self.env_suspect = 0
        self.env_sustain_samples = 4
        self.stable_suspect = 0
        self.stable_sustain_samples = 4

        # TRANSITION gating
        self.transition_settle_count = 0
        self.transition_settle_required = 6
        self.ambient_step_cap_c = 0.40

        # Exposed latest values for key commands
        self.last_press_hpa: Optional[float] = None
        self.last_raw_t_c: Optional[float] = None
        self.last_thrust: Optional[int] = None

        # User zero reference (pressure altitude)
        self.ground_alt_m = 0.0

    def get_state_scaling(self, thrust: int) -> float:
        if thrust == 0:
            self.state = "IDLE"
            return 1.0
        elif 0 < thrust < 35000:
            self.state = "FLIGHT (COOLING)"
            return 0.62
        else:
            self.state = "FLIGHT (POWER)"
            return 0.78

    def _enter_transition(self, reason: str):
        self.mode = "TRANSITION"
        self.env_suspect = 0
        self.stable_suspect = 0
        self.transition_settle_count = 0
        self.last_event = f"ENV CHANGE → TRANSITION ({reason})"

    def _exit_transition(self):
        self.mode = "STABLE"
        self.env_suspect = 0
        self.stable_suspect = 0
        self.transition_settle_count = 0
        self.last_event = "REBASelined → STABLE"

    def process(self, raw_t_c: float, thrust: int, press_hpa: float):
        self.last_raw_t_c = raw_t_c
        self.last_thrust = thrust
        self.last_press_hpa = press_hpa

        curr_time = time.time()
        dt = max(curr_time - self.last_update_time, 1e-3)
        self.last_update_time = curr_time

        # EMA and ROC
        self.ema_temp_c = (self.alpha * raw_t_c) + (1.0 - self.alpha) * self.ema_temp_c
        roc_c_s = (self.ema_temp_c - self.prev_ema_temp_c) / dt
        self.prev_ema_temp_c = self.ema_temp_c

        scaling = self.get_state_scaling(thrust)

        expected_raw_c = self.ambient_est_c + (self.idle_offset_c * scaling)
        deviation_c = self.ema_temp_c - expected_raw_c

        corr_t_c = self.ema_temp_c - (self.idle_offset_c * scaling)

        if thrust == 0:
            near_expected = abs(deviation_c) <= self.drift_band_c
            settled = abs(roc_c_s) <= self.roc_stable_thresh

            force_env_change = self.ema_temp_c <= (self.ambient_est_c - self.ambient_cool_margin_c)
            soft_env_change = abs(deviation_c) >= self.env_error_c

            if self.mode == "STABLE":
                if force_env_change or soft_env_change:
                    self.env_suspect += 1
                    self.last_event = "ENV CHANGE SUSPECTED"
                    if self.env_suspect >= self.env_sustain_samples:
                        self._enter_transition("force" if force_env_change else "deviation")
                else:
                    self.env_suspect = 0

                    slow_heating = 0.0 < roc_c_s < self.heat_thresh
                    slow_cooling = roc_c_s < 0.0 and abs(roc_c_s) < self.cool_thresh
                    if near_expected and (slow_heating or slow_cooling):
                        self.idle_offset_c += deviation_c * self.offset_gain
                        self.last_event = "Offset drift update (STABLE)"
                    else:
                        self.last_event = "Stable tracking"

                    if settled and near_expected:
                        self.ambient_est_c += (corr_t_c - self.ambient_est_c) * self.ambient_gain_stable

            else:  # TRANSITION
                if settled:
                    self.transition_settle_count += 1
                else:
                    self.transition_settle_count = 0

                if self.transition_settle_count < self.transition_settle_required:
                    self.stable_suspect = 0
                    self.last_event = (
                        f"TRANSITION waiting for settle ({self.transition_settle_count}/{self.transition_settle_required})"
                    )
                else:
                    step = (corr_t_c - self.ambient_est_c) * self.ambient_gain_transition
                    step = max(-self.ambient_step_cap_c, min(self.ambient_step_cap_c, step))
                    self.ambient_est_c += step

                    expected_raw_c = self.ambient_est_c + self.idle_offset_c
                    deviation_c = self.ema_temp_c - expected_raw_c

                    if abs(deviation_c) <= self.drift_band_c and settled:
                        self.stable_suspect += 1
                        self.last_event = f"TRANSITION settling ({self.stable_suspect}/{self.stable_sustain_samples})"
                        if self.stable_suspect >= self.stable_sustain_samples:
                            self._exit_transition()
                    else:
                        self.stable_suspect = 0
                        self.last_event = "TRANSITION (re-basing ambient)"

        abs_alt_m = 44330.0 * (1.0 - math.pow(press_hpa / P_STANDARD, 0.1903))
        density = (press_hpa * 100.0) / (R_DRY_AIR * (corr_t_c + 273.15))
        qff_hpa = press_hpa * math.pow(
            1.0 - (0.0065 * STATION_ELEV_M) / (corr_t_c + 0.0065 * STATION_ELEV_M + 273.15),
            -5.257,
        )

        return corr_t_c, roc_c_s, abs_alt_m, density, qff_hpa, expected_raw_c, deviation_c


def _detect_baro_varnames(scf: SyncCrazyflie) -> tuple[str, str]:
    candidates = [
        ("baro.temp", "baro.pressure"),
        ("bmp388.temp", "bmp388.pressure"),
        ("bmp3.temp", "bmp3.pressure"),
    ]

    toc = scf.cf.log.toc
    for t, p in candidates:
        if t in toc.toc and p in toc.toc:
            return t, p
    return candidates[0]

# ============================================================
#                   FLIGHT SUPPORT
# ============================================================
@dataclass
class Pose:
    x: float
    y: float
    z: float

class StateEstimate:
    def __init__(self):
        self.pose = Pose(0.0, 0.0, 0.0)
        self._updated = Event()

    def update(self, x, y, z):
        self.pose = Pose(x, y, z)
        self._updated.set()

    def wait_for_first(self, timeout_s: float) -> bool:
        return self._updated.wait(timeout_s)

def start_state_logging(cf: Crazyflie, state: StateEstimate) -> LogConfig:
    lg = LogConfig(name="StateEstimate", period_in_ms=STATE_LOG_PERIOD_MS)
    lg.add_variable("stateEstimate.x", "float")
    lg.add_variable("stateEstimate.y", "float")
    lg.add_variable("stateEstimate.z", "float")

    def _cb(ts, data, logconf):
        state.update(
            data["stateEstimate.x"],
            data["stateEstimate.y"],
            data["stateEstimate.z"],
        )

    cf.log.add_config(lg)
    lg.data_received_cb.add_callback(_cb)
    lg.start()
    return lg

def configure_cf(cf: Crazyflie):
    cf.param.set_value("stabilizer.estimator", "2")      # Kalman
    cf.param.set_value("commander.enHighLevel", "1")     # High-level commander
    try:
        cf.param.set_value("stabilizer.controller", "2") # Mellinger (if available)
    except Exception:
        pass

def reset_estimator(cf: Crazyflie):
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(2.0)

def compute_duration(a: Pose, b: Pose, speed_mps: float) -> float:
    d = math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)
    return max(0.8, d / max(0.05, speed_mps))

def go_to_abs_with_dwell(hl, state: StateEstimate, target: Pose, yaw: float = 0.0):
    """
    Go to an absolute position, wait until within POSITION_TOL_M, then dwell DWELL_TIME_S.
    """
    current = state.pose
    dur = compute_duration(current, target, SPEED_MPS)
    hl.go_to(target.x, target.y, target.z, yaw=yaw, duration_s=dur, relative=False)

    start_time = time.time()
    timeout = dur + 3.0
    while time.time() - start_time < timeout:
        p = state.pose
        d = math.sqrt((p.x - target.x) ** 2 + (p.y - target.y) ** 2 + (p.z - target.z) ** 2)
        if d < POSITION_TOL_M:
            break
        time.sleep(0.05)

    time.sleep(DWELL_TIME_S)

# Added Go to abs with dwell for geofencing and safety speeds

def go_to_abs_with_dwell_geofence_safe(
    hl,
    state: StateEstimate,
    target: Pose,
    fence: dict,
    recovery_started: Event,
    yaw: float = 0.0,
):
    if recovery_started.is_set():
        raise MissionAbort("Recovery in progress.")

    current = state.pose

    limited_speed = geofence_limited_speed(current, fence)

    if limited_speed <= 0.0:
        raise MissionAbort("Drone outside geofence.")

    dur = compute_duration(current, target, limited_speed)

    hl.go_to(target.x, target.y, target.z, yaw=yaw, duration_s=dur, relative=False)

    start_time = time.time()
    timeout = dur + 3.0

    while time.time() - start_time < timeout:
        if recovery_started.is_set():
            raise MissionAbort("Recovery triggered mid-flight.")

        p = state.pose

        if not pose_within_geofence(p, fence):
            raise MissionAbort("Drone exited geofence during movement.")

        d = math.sqrt((p.x - target.x)**2 + (p.y - target.y)**2 + (p.z - target.z)**2)
        if d < POSITION_TOL_M:
            break

        time.sleep(0.05)

    time.sleep(DWELL_TIME_S)

def generate_serpentine_waypoints(center: Pose, z: float, square_side_ft: float, step_ft: float):
    half = ft_to_m(square_side_ft / 2.0)
    step = ft_to_m(step_ft)

    x_sign = -1.0 if LEFT_IS_NEG_X else 1.0
    y_sign = -1.0 if BACK_IS_NEG_Y else 1.0

    x_left = center.x + x_sign * half
    x_right = center.x - x_sign * half
    y_back = center.y + y_sign * half
    y_front = center.y - y_sign * half

    lanes = int(round((2.0 * half) / step)) + 1

    waypoints = []
    for i in range(lanes):
        x = x_left if lanes == 1 else x_left + (x_right - x_left) * (i / (lanes - 1))

        if i % 2 == 0:
            y0, y1 = y_back, y_front
        else:
            y0, y1 = y_front, y_back

        waypoints.append(Pose(x, y0, z))

        lane_len = abs(y1 - y0)
        segs = max(1, int(round(lane_len / step)))
        for j in range(1, segs + 1):
            y = y0 + (y1 - y0) * (j / segs)
            waypoints.append(Pose(x, y, z))

    return waypoints

def fly_scan_layer(cf: Crazyflie, state: StateEstimate, center: Pose, altitude_ft: float, layer_name: str):
    hl = cf.high_level_commander
    z_target = ft_to_m(altitude_ft)

    if z_target > MAX_Z_M:
        raise ValueError(f"Requested altitude {altitude_ft} ft exceeds safety max.")

    print(f"\n=== {layer_name}: Takeoff to {altitude_ft:.2f} ft ({z_target:.2f} m) ===")
    hl.takeoff(z_target, TAKEOFF_TIME_S)
    time.sleep(TAKEOFF_TIME_S + 0.2)

    wps = generate_serpentine_waypoints(center, z_target, SQUARE_SIDE_FT, STEP_FT)

    print(f"{layer_name}: Move to back-left start corner...")
    go_to_abs_with_dwell(hl, state, wps[0])

    print(f"{layer_name}: Scanning {SQUARE_SIDE_FT}ft x {SQUARE_SIDE_FT}ft "
          f"with {STEP_FT}ft steps ({len(wps)} waypoints), dwell={DWELL_TIME_S:.1f}s...")
    for wp in wps:
        go_to_abs_with_dwell(hl, state, wp)

    print(f"{layer_name}: Return to center at same altitude...")
    go_to_abs_with_dwell(hl, state, Pose(center.x, center.y, z_target))

# ============================================================
#          SENSOR LOGGING SETUP + START/STOP
# ============================================================
class SensorLogger:
    """
    Starts your sensor logging (same calculations + same CSV columns),
    and can be stopped cleanly after landing.
    """
    def __init__(self, scf: SyncCrazyflie):
        self.scf = scf
        self.csv_f = None
        self.writer = None
        self.full_log = None

        self.engine: Optional[ThermalEngine] = None
        self.temp_var = ""
        self.press_var = ""
        self.temp_unit = "C"

        self.rows_since_flush = 0
        self.last_print = 0.0
        self.sample_index = 0

    def start(self):
        # Detect baro variables
        self.temp_var, self.press_var = _detect_baro_varnames(self.scf)

        # Pull initial reading
        init_log = LogConfig(name="Init", period_in_ms=100)
        init_log.add_variable(self.temp_var, "float")
        init_log.add_variable(self.press_var, "float")

        i_raw_c = None
        i_press = None
        with SyncLogger(self.scf, init_log) as logger:
            for entry in logger:
                i_raw_c = entry[1][self.temp_var]
                i_press = entry[1][self.press_var]
                break

        if i_raw_c is None or i_press is None:
            raise RuntimeError("Failed to read initial barometer values.")

        print("\n--- SESSION INITIALIZATION (Sensor Logging) ---")
        print(f"Baro vars: temp='{self.temp_var}' press='{self.press_var}'")
        print(f"Startup Temp: {i_raw_c:.2f}°C / {c_to_f(i_raw_c):.2f}°F")

        user_in = input("Ambient Room Temp (e.g. 74F or 22C): ").strip()
        ambient_c, user_temp_unit = parse_temp_user(user_in)
        self.temp_unit = user_temp_unit

        self.engine = ThermalEngine(initial_raw_c=i_raw_c, initial_ambient_c=ambient_c)
        self.engine.ground_alt_m = 44330.0 * (1.0 - math.pow(i_press / P_STANDARD, 0.1903))

        # Open CSV ONCE
        self.csv_f = open(CSV_FILE, mode="w", newline="", encoding="utf-8-sig")
        self.writer = csv.writer(self.csv_f)

        def to_user_units(c_val: float) -> float:
            return c_to_f(c_val) if self.temp_unit == "F" else c_val

        def roc_to_user_units(roc_c_s: float) -> float:
            return (roc_c_s * 9.0 / 5.0) if self.temp_unit == "F" else roc_c_s

        # Header row (UNCHANGED fields)
        self.writer.writerow([
            "index",
            "datetime_et",
            "state",
            "mode",
            "event",
            f"roc_{self.temp_unit}_per_s",
            f"raw_temp_{self.temp_unit}",
            f"expected_raw_{self.temp_unit}",
            f"deviation_{self.temp_unit}",
            f"corr_temp_{self.temp_unit}",
            f"thermal_offset_{self.temp_unit}",
            f"ambient_est_{self.temp_unit}",
            "rel_alt_m",
            "rel_alt_ft",
            "air_density_kg_m3",
            "station_pressure_hPa",
            "sea_level_pressure_QFF_hPa",
            "battery_V",
            "thrust",
        ])
        self.csv_f.flush()

        # Log callback (UNCHANGED row contents)
        def data_callback(timestamp_ms, data, logconf):
            if self.engine is None or self.writer is None:
                return

            raw_t_c = data.get(self.temp_var)
            press_hpa = data.get(self.press_var)
            thrust = int(data.get("stabilizer.thrust", 0))
            vbat = data.get("pm.vbat")

            if raw_t_c is None or press_hpa is None:
                return

            corr_t_c, roc_c_s, abs_alt_m, density, qff_hpa, expected_raw_c, deviation_c = \
                self.engine.process(raw_t_c, thrust, press_hpa)

            rel_alt_m = abs_alt_m - self.engine.ground_alt_m

            raw_t_f = c_to_f(raw_t_c)
            corr_t_f = c_to_f(corr_t_c)
            roc_f_s = roc_c_s * 9.0 / 5.0

            now_s = time.time()
            if (now_s - self.last_print) >= PRINT_EVERY_S:
                self.last_print = now_s
                print(
                    f"\n{'='*22} System Status [{timestamp_ms}ms] {'='*22}\n"
                    f"STATUS:  {self.engine.state} | {self.engine.last_event}\nMODE:    {self.engine.mode} | AmbientEst: {self.engine.ambient_est_c:.2f}°C / {c_to_f(self.engine.ambient_est_c):.2f}°F\n"
                    f"ROC:     {roc_c_s:+.4f} °C/s | {roc_f_s:+.4f} °F/s\n\n"
                    f"TEMPERATURE:\n"
                    f"  Raw:         {raw_t_c:.2f}°C / {raw_t_f:.2f}°F\n"
                    f"  ExpectedRaw: {expected_raw_c:.2f}°C / {c_to_f(expected_raw_c):.2f}°F (ambient+offset)\n"
                    f"  Corrected:   {corr_t_c:.2f}°C / {corr_t_f:.2f}°F\n"
                    f"  Deviation:   {deviation_c:+.3f} °C (raw-expected)\n"
                    f"  Offset:      {self.engine.idle_offset_c:.4f} °C\n\n"
                    f"ALTITUDE & AIR:\n"
                    f"  Relative:  {rel_alt_m:.2f} m / {m_to_ft(rel_alt_m):.2f} ft (AGL)\n"
                    f"  Density:   {density:.4f} kg/m³\n\n"
                    f"PRESSURE & POWER:\n"
                    f"  Station P: {press_hpa:.2f} hPa | Sea Level (QFF): {qff_hpa:.2f} hPa\n"
                    f"  Battery:   {vbat if vbat is not None else float('nan'):.2f} V | Thrust: {thrust}\n"
                    f"{'='*65}\n"
                )

            self.sample_index += 1
            dt_et = datetime.now().astimezone().isoformat(timespec="milliseconds")

            roc_u_s = roc_to_user_units(roc_c_s)
            raw_t_u = to_user_units(raw_t_c)
            expected_raw_u = to_user_units(expected_raw_c)
            deviation_u = to_user_units(deviation_c)
            corr_t_u = to_user_units(corr_t_c)
            offset_u = to_user_units(self.engine.idle_offset_c)
            ambient_est_u = to_user_units(self.engine.ambient_est_c)

            self.writer.writerow([
                self.sample_index,
                dt_et,
                self.engine.state,
                self.engine.mode,
                self.engine.last_event,
                roc_u_s,
                raw_t_u,
                expected_raw_u,
                deviation_u,
                corr_t_u,
                offset_u,
                ambient_est_u,
                rel_alt_m,
                m_to_ft(rel_alt_m),
                density,
                press_hpa,
                qff_hpa,
                vbat,
                thrust,
            ])

            self.rows_since_flush += 1
            if self.rows_since_flush >= CSV_FLUSH_EVERY and self.csv_f is not None:
                self.csv_f.flush()
                self.rows_since_flush = 0

        # Start the log config (same vars)
        self.full_log = LogConfig(name="Master", period_in_ms=LOG_PERIOD_MS)
        self.full_log.add_variable(self.temp_var, "float")
        self.full_log.add_variable(self.press_var, "float")
        self.full_log.add_variable("pm.vbat", "float")
        self.full_log.add_variable("stabilizer.thrust", "uint16_t")

        self.scf.cf.log.add_config(self.full_log)
        self.full_log.data_received_cb.add_callback(data_callback)
        self.full_log.start()

        print("Sensor logging started (will stop automatically after landing).")

    def stop(self):
        if self.full_log is not None:
            try:
                self.full_log.stop()
            except Exception:
                pass

        if self.csv_f is not None:
            try:
                self.csv_f.flush()
            finally:
                self.csv_f.close()
                self.csv_f = None

        print(f"CSV saved to: {CSV_FILE}")

# ============================================================
#                         MAIN
# ============================================================
def main():
    print("Autonomous scan + sensor logging\n")
    print(f"Square: {SQUARE_SIDE_FT} ft | Step: {STEP_FT} ft | Layers: {ALT_LAYERS_FT} ft")
    print(f"Dwell per waypoint: {DWELL_TIME_S:.1f}s | Speed: {SPEED_MPS} m/s\n")

    cflib.crtp.init_drivers(enable_debug_driver=False)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf
        configure_cf(cf)

        # Position state logging
        state = StateEstimate()
        pos_lg = start_state_logging(cf, state)

        if not state.wait_for_first(WAIT_FOR_ESTIMATE_TIMEOUT_S):
            pos_lg.stop()
            raise RuntimeError("No stateEstimate received. Check Lighthouse/Kalman.")

        print("Keep the drone still. Resetting estimator...")
        reset_estimator(cf)
        time.sleep(1.0)

        # Capture origin from estimator
        origin = state.pose
        print("Captured origin (center) from estimator:")
        print(f"  x0={origin.x:.3f}, y0={origin.y:.3f}, z0={origin.z:.3f}")

        # Build 4x4m geofence centered on takeoff origin
        fence = build_geofence_from_origin(origin)
        print("\nGeofence enabled:")
        print(f"  X: [{fence['x_min']:.3f}, {fence['x_max']:.3f}]")
        print(f"  Y: [{fence['y_min']:.3f}, {fence['y_max']:.3f}]")
        print(f"  Z: [{fence['z_min']:.3f}, {fence['z_max']:.3f}]")

        # Start sensor logging before takeoff
        sensor_logger = SensorLogger(scf)
        sensor_logger.start()

        # Start background geofence monitor
        geofence_stop = Event()
        recovery_started = Event()
        monitor_thread = Thread(
            target=geofence_monitor_loop,
            args=(cf, state, origin, fence, geofence_stop, recovery_started),
            daemon=True
        )
        monitor_thread.start()

        try:
            for idx, alt_ft in enumerate(ALT_LAYERS_FT, start=1):
                if recovery_started.is_set():
                    raise MissionAbort("Geofence recovery already triggered.")

                fly_scan_layer(cf, state, center=origin, altitude_ft=alt_ft, layer_name=f"LAYER {idx}")

                if recovery_started.is_set():
                    raise MissionAbort("Geofence recovery triggered during scan.")

            print("\nAll layers complete. Landing...")

        except MissionAbort as e:
            print(f"\nABORT: {e}")

        except KeyboardInterrupt:
            print("\nABORT: KeyboardInterrupt. Landing...")

        finally:
            geofence_stop.set()
            monitor_thread.join(timeout=1.0)

            # Only do normal landing if failsafe did not already take over
            if not recovery_started.is_set():
                try:
                    cf.high_level_commander.land(0.0, LAND_TIME_S)
                    time.sleep(LAND_TIME_S + 0.3)
                    cf.high_level_commander.stop()
                except Exception:
                    pass

            # Stop sensor logging and save CSV
            sensor_logger.stop()

            # Stop position logging
            try:
                pos_lg.stop()
            except Exception:
                pass

            print("Done.")