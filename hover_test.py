import time
import sys
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")

HOVER_SECONDS = 10.0
COMMAND_HZ = 10.0  # re-issue go_to at 10 Hz
TAKEOFF_TIME = 2.0
LAND_TIME = 2.5
WAIT_FOR_ESTIMATE_TIMEOUT = 8.0


def feet_to_meters(ft: float) -> float:
    return ft * 0.3048


class StateEstimate:
    """Simple holder for stateEstimate logs."""
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self._updated = Event()

    def update(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self._updated.set()

    def wait_for_first(self, timeout_s: float) -> bool:
        return self._updated.wait(timeout_s)


def reset_estimator(cf: Crazyflie):
    # Reset Kalman (recommended after enabling lighthouse / moving base station)
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(2.0)


def configure_for_position_hold(cf: Crazyflie):
    # Ensure Kalman estimator
    cf.param.set_value("stabilizer.estimator", "2")   # 2 = Kalman
    # Use Mellinger controller if available (often best for position control)
    # If your firmware uses different values, this won't break anything; it just may be ignored.
    try:
        cf.param.set_value("stabilizer.controller", "2")  # 2 = Mellinger
    except Exception:
        pass
    # Enable high-level commander
    cf.param.set_value("commander.enHighLevel", "1")


def start_state_logging(cf: Crazyflie, state: StateEstimate) -> LogConfig:
    lg = LogConfig(name="StateEstimate", period_in_ms=50)  # 20 Hz logs
    lg.add_variable("stateEstimate.x", "float")
    lg.add_variable("stateEstimate.y", "float")
    lg.add_variable("stateEstimate.z", "float")

    def _log_cb(timestamp, data, logconf):
        state.update(
            data["stateEstimate.x"],
            data["stateEstimate.y"],
            data["stateEstimate.z"],
        )

    cf.log.add_config(lg)
    lg.data_received_cb.add_callback(_log_cb)
    lg.start()
    return lg


def main():
    try:
        altitude_ft = float(input("Enter hover altitude (feet): ").strip())
    except ValueError:
        print("Invalid number.")
        sys.exit(1)

    z_target = feet_to_meters(altitude_ft)
    # Keep it sane for lab testing
    if z_target < 0.15:
        print("Altitude too low; set at least 0.5 ft (~0.15 m).")
        sys.exit(1)

    print(f"Target altitude: {z_target:.2f} m")

    cflib.crtp.init_drivers(enable_debug_driver=False)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf

        configure_for_position_hold(cf)

        state = StateEstimate()
        lg = start_state_logging(cf, state)

        # Make sure we have an estimate before continuing
        if not state.wait_for_first(WAIT_FOR_ESTIMATE_TIMEOUT):
            lg.stop()
            raise RuntimeError("No stateEstimate received. Is the estimator running?")

        # Reset estimator once we’re logging and stationary
        print("Resetting estimator (keep the drone still)...")
        reset_estimator(cf)

        # Wait a moment for it to settle
        time.sleep(1.0)

        # Capture the "anchor" position
        x0, y0 = state.x, state.y
        print(f"Anchor position locked: x0={x0:.3f}, y0={y0:.3f}")

        hl = cf.high_level_commander

        # Take off
        print("Taking off...")
        hl.takeoff(z_target, TAKEOFF_TIME)
        time.sleep(TAKEOFF_TIME + 0.3)

        # XY-locked hover loop
        print(f"Holding XY at (x0, y0) for {HOVER_SECONDS:.0f}s...")
        dt = 1.0 / COMMAND_HZ
        t_end = time.time() + HOVER_SECONDS

        try:
            while time.time() < t_end:
                # Keep commanding the same absolute position (x0,y0,z_target).
                # yaw=0.0 keeps yaw fixed; you can change if you want.
                hl.go_to(x0, y0, z_target, yaw=0.0, duration_s=dt, relative=False)
                time.sleep(dt)
        except KeyboardInterrupt:
            print("\nAbort requested, landing...")

        # Land
        print("Landing...")
        hl.land(0.0, LAND_TIME)
        time.sleep(LAND_TIME + 0.3)
        hl.stop()

        lg.stop()
        print("Done.")


if __name__ == "__main__":
    main()