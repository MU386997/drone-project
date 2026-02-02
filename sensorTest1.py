import time
import sys
import select
import os

IS_WINDOWS = os.name == 'nt'
if IS_WINDOWS:
    import msvcrt
import math
import csv
from typing import Optional

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger

# --- CONFIGURATION ---
URI = "radio://0/80/2M"

P_STANDARD = 1013.25  # hPa
R_DRY_AIR = 287.05    # J/(kg*K)
STATION_ELEV_M = 15   # Alexandria, VA (for QFF reduction)

LOG_PERIOD_MS = 500   # 2 Hz
CSV_FILE = "drone_mapping_data.csv"  # keep your original name
CSV_FLUSH_EVERY = 20
# Print cadence (terminal). Set to 0.0 to print every log sample.
# User request: print every reading (LOG_PERIOD_MS=500ms => 2 Hz).
PRINT_EVERY_S = 0.0

# --- HELPERS ---
def c_to_f(c: float) -> float:
    return (c * 9.0 / 5.0) + 32.0


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def m_to_ft(m: float) -> float:
    return m * 3.28084


def parse_temp_user(s: str) -> float:
    """Parse ambient temperature like '74F' or '22C' (case-insensitive).
    Returns temperature in Celsius.
    """
    if s is None:
        raise ValueError("empty")
    s = s.strip().upper().replace(" ", "")
    if not s:
        raise ValueError("empty")

    # Allow raw number with no suffix -> assume C
    if s[-1] not in ("C", "F"):
        return float(s)

    unit = s[-1]
    val = float(s[:-1])
    return f_to_c(val) if unit == "F" else val


class ThermalEngine:
    """Thermal correction engine with two-timescale tracking.

    Goals:
      - Track *ambient* temperature changes (even small ones) so corrected output follows reality.
      - Track CPU/self-heating bias separately and conservatively so we don't 'correct away' real cooling/heating.

    Key state:
      - ambient_est_c: current estimate of ambient temperature (allowed to move).
      - idle_offset_c: estimate of CPU heat bias at idle (moves slowly, only when safe).
      - mode: STABLE vs TRANSITION (environment change / re-basing period).
    """  # noqa: E501

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
        self.heat_thresh = 0.07          # °C/s (slow heating that might be CPU drift)
        self.cool_thresh = 0.03          # °C/s (slow cooling that might be CPU drift)
        self.roc_stable_thresh = 0.02    # °C/s (consider 'settled')
        self.drift_band_c = 0.45         # °C (how close to expected_raw to allow offset adaptation)
        self.env_error_c = 1.00          # °C (deviation from expected_raw that suggests environment change)
        self.ambient_cool_margin_c = 0.30  # °C (if raw drops below ambient_est by this => definitely cooled environment)

        # Gains (per-sample-ish; actual behavior depends on log rate)
        self.offset_gain = 0.05          # how quickly idle_offset adapts (slow)
        self.ambient_gain_stable = 0.03  # how quickly ambient_est follows corrected temp in stable conditions
        self.ambient_gain_transition = 0.15  # faster re-basing when environment has changed

        # Counters for hysteresis (avoid flapping)
        self.env_suspect = 0
        self.env_sustain_samples = 4      # 4 samples @ 2Hz ~= 2s
        self.stable_suspect = 0
        self.stable_sustain_samples = 4   # 2s of settled behavior to exit TRANSITION

        # TRANSITION gating:
        # When you go from a very cold environment -> warm room, the raw reading rises rapidly.
        # If we immediately re-base ambient_est during this rapid transient, we can latch a
        # "too-cold" ambient (e.g., ~63°F) before the sensor has stabilized. Instead, we
        # wait for the raw temperature rate-of-change (ROC) to settle for several samples
        # before allowing ambient_est to move.
        self.transition_settle_count = 0
        self.transition_settle_required = 6   # 6 samples @ 2Hz ~= 3s settled before re-basing

        # Once settled, we still cap how fast ambient_est can move per sample to prevent
        # overshoot / latching from a single noisy point.
        self.ambient_step_cap_c = 0.40        # °C per sample max change while re-basing

        # Exposed latest values for key commands
        self.last_press_hpa: Optional[float] = None
        self.last_raw_t_c: Optional[float] = None
        self.last_thrust: Optional[int] = None

        # User zero reference (pressure altitude)
        self.ground_alt_m = 0.0

    def get_state_scaling(self, thrust: int) -> float:
        # Keep your earlier structure; currently we only *adapt* estimates at thrust==0.
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
        # Store latest values for key commands
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

        # Expected raw temperature given our current ambient estimate + cpu bias model
        expected_raw_c = self.ambient_est_c + (self.idle_offset_c * scaling)
        deviation_c = self.ema_temp_c - expected_raw_c

        # Base corrected temperature (no predictive term; stable for idle validation)
        corr_t_c = self.ema_temp_c - (self.idle_offset_c * scaling)

        # Only adapt the estimates at idle. In flight, we still output corrected values,
        # but we don't want to 'learn' offset/ambient from turbulence/propwash yet.
        if thrust == 0:
            near_expected = abs(deviation_c) <= self.drift_band_c
            settled = abs(roc_c_s) <= self.roc_stable_thresh

            # Hard physics rule: if measured raw drops below our ambient estimate by margin,
            # the environment got colder (CPU heat cannot cause this direction).
            force_env_change = self.ema_temp_c <= (self.ambient_est_c - self.ambient_cool_margin_c)

            # Soft env-change indicator: large deviation from expected_raw
            soft_env_change = abs(deviation_c) >= self.env_error_c

            if self.mode == "STABLE":
                # Require env-change to persist for a few samples to avoid false triggers.
                if force_env_change or soft_env_change:
                    self.env_suspect += 1
                    self.last_event = "ENV CHANGE SUSPECTED"
                    if self.env_suspect >= self.env_sustain_samples:
                        self._enter_transition("force" if force_env_change else "deviation")
                else:
                    self.env_suspect = 0

                    # 1) Slow CPU-bias adaptation (ONLY when the model is consistent)
                    slow_heating = 0.0 < roc_c_s < self.heat_thresh
                    slow_cooling = roc_c_s < 0.0 and abs(roc_c_s) < self.cool_thresh
                    if near_expected and (slow_heating or slow_cooling):
                        # Deviation here is mainly CPU bias error; update offset slowly.
                        self.idle_offset_c += deviation_c * self.offset_gain
                        self.last_event = "Offset drift update (STABLE)"
                    else:
                        self.last_event = "Stable tracking"

                    # 2) Ambient estimate adaptation (this is what lets you track small changes)
                    # Only do this when we're settled and near expected (to avoid chasing transient noise).
                    if settled and near_expected:
                        # corr_t_c is our best estimate of ambient at this instant
                        self.ambient_est_c += (corr_t_c - self.ambient_est_c) * self.ambient_gain_stable

            else:  # TRANSITION
                # Freeze offset; re-base ambient estimate toward the new environment.
                # BUT only after the raw temperature has *settled* (ROC small for multiple samples).

                if settled:
                    self.transition_settle_count += 1
                else:
                    self.transition_settle_count = 0

                if self.transition_settle_count < self.transition_settle_required:
                    # During rapid warm/cool transients, do NOT move ambient_est.
                    # Otherwise we can lock onto an intermediate temperature.
                    self.stable_suspect = 0
                    self.last_event = (
                        f"TRANSITION waiting for settle ({self.transition_settle_count}/{self.transition_settle_required})"
                    )
                else:
                    # Now it's safe to move ambient_est toward our best ambient guess (corr_t_c).
                    step = (corr_t_c - self.ambient_est_c) * self.ambient_gain_transition
                    # Cap per-sample step to avoid single-sample noise driving the estimate.
                    step = max(-self.ambient_step_cap_c, min(self.ambient_step_cap_c, step))
                    self.ambient_est_c += step

                    # Recompute expected/deviation after ambient update (for exit logic)
                    expected_raw_c = self.ambient_est_c + self.idle_offset_c  # scaling==1 at idle
                    deviation_c = self.ema_temp_c - expected_raw_c

                    # Exit TRANSITION after sustained settled behavior with small deviation.
                    if abs(deviation_c) <= self.drift_band_c and settled:
                        self.stable_suspect += 1
                        self.last_event = f"TRANSITION settling ({self.stable_suspect}/{self.stable_sustain_samples})"
                        if self.stable_suspect >= self.stable_sustain_samples:
                            self._exit_transition()
                    else:
                        self.stable_suspect = 0
                        self.last_event = "TRANSITION (re-basing ambient)"

        # Derived metrics
        abs_alt_m = 44330.0 * (1.0 - math.pow(press_hpa / P_STANDARD, 0.1903))
        density = (press_hpa * 100.0) / (R_DRY_AIR * (corr_t_c + 273.15))
        qff_hpa = press_hpa * math.pow(
            1.0 - (0.0065 * STATION_ELEV_M) / (corr_t_c + 0.0065 * STATION_ELEV_M + 273.15),
            -5.257,
        )

        return corr_t_c, roc_c_s, abs_alt_m, density, qff_hpa, expected_raw_c, deviation_c


def _detect_baro_varnames(scf: SyncCrazyflie) -> tuple[str, str]:
    """Return (temp_var, press_var) that exist in the TOC.

    Your original script uses 'baro.temp' and 'baro.pressure'.
    Some firmwares/sensors expose other names, so we try a short list.
    """
    candidates = [
        ("baro.temp", "baro.pressure"),
        ("bmp388.temp", "bmp388.pressure"),
        ("bmp3.temp", "bmp3.pressure"),
    ]

    toc = scf.cf.log.toc
    for t, p in candidates:
        if t in toc.toc and p in toc.toc:
            return t, p
    return candidates[0]  # fallback; you'll get a clear error if it truly doesn't exist


def run_master_engine():
    cflib.crtp.init_drivers()

    print("Connecting to Crazyflie...")
    csv_f = None
    try:
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
            # Detect the right baro variables (prevents your bmp388.temp TOC crash)
            temp_var, press_var = _detect_baro_varnames(scf)

            # Pull one initial reading for baseline + ground altitude
            init_log = LogConfig(name="Init", period_in_ms=100)
            init_log.add_variable(temp_var, "float")
            init_log.add_variable(press_var, "float")

            with SyncLogger(scf, init_log) as logger:
                i_raw_c = None
                i_press = None
                for entry in logger:
                    i_raw_c = entry[1][temp_var]
                    i_press = entry[1][press_var]
                    break

            if i_raw_c is None or i_press is None:
                print("Failed to read initial barometer values. Exiting.")
                return

            print("\n--- SESSION INITIALIZATION ---")
            print(f"Baro vars: temp='{temp_var}' press='{press_var}'")
            print(f"Startup Temp: {i_raw_c:.2f}°C / {c_to_f(i_raw_c):.2f}°F")

            # Ambient input (keep your old UX)
            user_in = input("Ambient Room Temp (e.g. 74F or 22C): ").strip()
            try:
                ambient_c = parse_temp_user(user_in)
            except Exception:
                print("Invalid ambient temperature. Use formats like 74F or 22C.")
                return

            engine = ThermalEngine(initial_raw_c=i_raw_c, initial_ambient_c=ambient_c)
            engine.ground_alt_m = 44330.0 * (1.0 - math.pow(i_press / P_STANDARD, 0.1903))

            # Open CSV ONCE
            csv_f = open(CSV_FILE, mode="w", newline="", encoding="utf-8-sig")
            writer = csv.writer(csv_f)
            writer.writerow([
                "timestamp_s", "state", "mode", "event", "roc_C_per_s", "roc_F_per_s",
                "raw_temp_C", "raw_temp_F", "expected_raw_C", "expected_raw_F", "deviation_C", "corr_temp_C", "corr_temp_F", "thermal_offset_C", "ambient_est_C", "ambient_est_F",
                "rel_alt_m", "rel_alt_ft", "air_density_kg_m3",
                "station_pressure_hPa", "sea_level_pressure_QFF_hPa", "battery_V", "thrust",
                "temp_var", "press_var",
            ])
            csv_f.flush()

            rows_since_flush = 0
            last_print = 0.0

            def data_callback(timestamp_ms, data, logconf):
                nonlocal rows_since_flush, last_print

                raw_t_c = data.get(temp_var)
                press_hpa = data.get(press_var)
                thrust = int(data.get("stabilizer.thrust", 0))
                vbat = data.get("pm.vbat")

                if raw_t_c is None or press_hpa is None:
                    return

                corr_t_c, roc_c_s, abs_alt_m, density, qff_hpa, expected_raw_c, deviation_c = engine.process(raw_t_c, thrust, press_hpa)
                rel_alt_m = abs_alt_m - engine.ground_alt_m

                raw_t_f = c_to_f(raw_t_c)
                corr_t_f = c_to_f(corr_t_c)
                roc_f_s = roc_c_s * 9.0 / 5.0

                now_s = time.time()
                if (now_s - last_print) >= PRINT_EVERY_S:
                    last_print = now_s
                    print(
                        f"\n{'='*22} System Status [{timestamp_ms}ms] {'='*22}\n"
                        f"STATUS:  {engine.state} | {engine.last_event}\nMODE:    {engine.mode} | AmbientEst: {engine.ambient_est_c:.2f}°C / {c_to_f(engine.ambient_est_c):.2f}°F\n"
                        f"ROC:     {roc_c_s:+.4f} °C/s | {roc_f_s:+.4f} °F/s\n\n"
                        f"TEMPERATURE:\n"
                        f"  Raw:         {raw_t_c:.2f}°C / {raw_t_f:.2f}°F\n"
                        f"  ExpectedRaw: {expected_raw_c:.2f}°C / {c_to_f(expected_raw_c):.2f}°F (ambient+offset)\n"
                        f"  Corrected:   {corr_t_c:.2f}°C / {corr_t_f:.2f}°F\n"
                        f"  Deviation:   {deviation_c:+.3f} °C (raw-expected)\n"
                        f"  Offset:      {engine.idle_offset_c:.4f} °C\n\n"
                        f"ALTITUDE & AIR:\n"
                        f"  Relative:  {rel_alt_m:.2f} m / {m_to_ft(rel_alt_m):.2f} ft (AGL)\n"
                        f"  Density:   {density:.4f} kg/m³\n\n"
                        f"PRESSURE & POWER:\n"
                        f"  Station P: {press_hpa:.2f} hPa | Sea Level (QFF): {qff_hpa:.2f} hPa\n"
                        f"  Battery:   {vbat if vbat is not None else float('nan'):.2f} V | Thrust: {thrust}\n"
                        f"{'='*65}\n"
                        "Keys: 't' = Temp Reset | 'z' = Zero Altitude | 'Ctrl+C' = Quit"
                    )

                writer.writerow([
                    now_s, engine.state, engine.mode, engine.last_event,
                    roc_c_s, roc_f_s,
                    raw_t_c, raw_t_f,
                    expected_raw_c, c_to_f(expected_raw_c), deviation_c,
                    corr_t_c, corr_t_f,
                    engine.idle_offset_c, engine.ambient_est_c, c_to_f(engine.ambient_est_c),
                    rel_alt_m, m_to_ft(rel_alt_m), density,
                    press_hpa, qff_hpa,
                    vbat, thrust,
                    temp_var, press_var,
                ])

                rows_since_flush += 1
                if rows_since_flush >= CSV_FLUSH_EVERY:
                    csv_f.flush()
                    rows_since_flush = 0

            # Full log configuration
            full_log = LogConfig(name="Master", period_in_ms=LOG_PERIOD_MS)
            full_log.add_variable(temp_var, "float")
            full_log.add_variable(press_var, "float")
            full_log.add_variable("pm.vbat", "float")
            full_log.add_variable("stabilizer.thrust", "uint16_t")

            scf.cf.log.add_config(full_log)
            full_log.data_received_cb.add_callback(data_callback)
            full_log.start()

            print("Logging started.")
            print("Commands: [t]=recalibrate idle offset, [z]=zero altitude, Ctrl+C=stop")

            try:
                while True:
                    key = None

                    if IS_WINDOWS:
                        # Windows: select() does not work on console stdin; use msvcrt.
                        if msvcrt.kbhit():
                            key = msvcrt.getwch()
                    else:
                        # macOS/Linux: non-blocking read from stdin via select().
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            key = sys.stdin.read(1)

                    if key:
                        key = key.lower()

                        if key == "t":
                            engine.idle_offset_c = engine.ema_temp_c - engine.ambient_est_c
                            engine.last_event = "Manual idle offset recalibration"

                        elif key == "z":
                            if engine.last_press_hpa is None:
                                print("No pressure reading yet; cannot zero altitude.")
                            else:
                                engine.ground_alt_m = 44330.0 * (
                                    1.0 - math.pow(engine.last_press_hpa / P_STANDARD, 0.1903)
                                )
                                engine.last_event = "Altitude zeroed (ground set)"
                                print(f"Ground altitude set to {engine.ground_alt_m:.2f} m (pressure altitude).")

                    time.sleep(0.05)

            except KeyboardInterrupt:
                print("\nStopping logging...")

            finally:
                full_log.stop()

    finally:
        if csv_f is not None:
            try:
                csv_f.flush()
            finally:
                csv_f.close()
        print(f"CSV saved to: {CSV_FILE}")


if __name__ == "__main__":
    run_master_engine()
