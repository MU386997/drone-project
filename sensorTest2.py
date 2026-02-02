import time
import sys
import select
import math
import msvcrt
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger

# --- CONFIGURATION ---
URI = 'radio://0/80/2M'
P_STANDARD = 1013.25 
R_DRY_AIR = 287.05
ELEV_OFFSET = 15 

def c_to_f(c): return (c * 9/5) + 32
def m_to_ft(m): return m * 3.28084
def f_to_c(f): return (f - 32) * 5/9

class ThermalEngine:
    def __init__(self, initial_raw, initial_ambient):
        # 1. Thermal Modeling & Filtering
        self.alpha = 0.12           
        self.tau = 3.2              
        self.ema_temp = initial_raw
        self.prev_ema_temp = initial_raw
        self.ambient_baseline = initial_ambient
        self.idle_offset = initial_raw - initial_ambient
        
        # 2. Thresholds
        self.heat_thresh = 0.07     
        self.cool_thresh = 0.03     
        
        # 3. Inflection & Trend Tracking
        self.trend_direction = 0     # 1 for increasing, -1 for decreasing
        self.trend_counter = 0       # How many samples has the new trend lasted?
        self.is_forcing_env = False  # Are we currently forcing env-change mode?
        
        # 4. State
        self.state = "IDLE"
        self.last_event = "Calibrated"
        self.lock_until = 0
        self.last_update_time = time.time()
        self.ground_alt_m = 0.0

    def get_state_scaling(self, thrust):
        if thrust == 0:
            self.state = "IDLE"
            return 1.0 
        elif 0 < thrust < 35000:
            self.state = "FLIGHT (COOLING)"
            return 0.62 
        else:
            self.state = "FLIGHT (POWER)"
            return 0.78 

    def process(self, raw_t, thrust, press):
        curr_time = time.time()
        dt = curr_time - self.last_update_time
        self.last_update_time = curr_time

        # A. Filter Update
        self.ema_temp = (self.alpha * raw_t) + (1 - self.alpha) * self.ema_temp
        roc = (self.ema_temp - self.prev_ema_temp) / dt if dt > 0 else 0
        self.prev_ema_temp = self.ema_temp
        
        # B. Inflection Detection Logic (The "Level Up")
        # Determine current instantaneous direction
        current_dir = 1 if roc > 0.005 else (-1 if roc < -0.005 else 0)
        
        if current_dir != 0 and current_dir != self.trend_direction:
            self.trend_counter += 1
            # If trend persists for 4 samples (approx 2 seconds)
            if self.trend_counter >= 4:
                self.trend_direction = current_dir
                self.trend_counter = 0
                # If we were cooling and now we are warming (or vice versa), force ENV mode
                self.is_forcing_env = True
                self.lock_until = curr_time + 3.0 
        else:
            self.trend_counter = 0

        # C. State Machine & Lockout Logic
        if thrust == 0:
            # Check if curve has flattened enough to exit forced ENV mode
            if self.is_forcing_env and abs(roc) < 0.01:
                self.is_forcing_env = False

            if curr_time < self.lock_until or self.is_forcing_env:
                self.last_event = "ENV CHANGE (Trend Shift)" if self.is_forcing_env else "ENV CHANGE (Velocity)"
            else:
                is_drift = False
                if roc > 0 and roc < self.heat_thresh: is_drift = True
                elif roc < 0 and abs(roc) < self.cool_thresh: is_drift = True
                
                if is_drift:
                    drift_error = self.ema_temp - (self.ambient_baseline + self.idle_offset)
                    self.idle_offset += drift_error * 0.07 
                    self.last_event = "Adjusting to Drift"
                else:
                    self.last_event = "ENV CHANGE (Threshold)"
                    self.lock_until = curr_time + 3.0
        
        # D. Predict & Calculate
        predicted_raw = self.ema_temp + (self.tau * roc)
        corr_t = predicted_raw - (self.idle_offset * self.get_state_scaling(thrust))
        
        abs_alt = 44330 * (1.0 - math.pow(press / P_STANDARD, 0.1903))
        density = (press * 100) / (R_DRY_AIR * (corr_t + 273.15))
        qff = press * math.pow(1 - (0.0065 * 15) / (corr_t + 0.0065 * 15 + 273.15), -5.257)
        
        return corr_t, roc, abs_alt, density, qff

def run_master_engine():
    cflib.crtp.init_drivers()
    
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache=None)) as scf:
        init_log = LogConfig(name='Init', period_in_ms=100)
        init_log.add_variable('baro.temp', 'float'); init_log.add_variable('baro.pressure', 'float')
        
        with SyncLogger(scf, init_log) as logger:
            for entry in logger:
                i_raw, i_press = entry[1]['baro.temp'], entry[1]['baro.pressure']
                break

        print(f"\n--- SESSION INITIALIZATION ---")
        print(f"Startup Temp: {i_raw:.2f}°C / {c_to_f(i_raw):.2f}°F")
        user_in = input("Ambient Room Temp (e.g. 74F or 22C): ").strip().upper()
        
        ambient = f_to_c(float(user_in[:-1])) if user_in.endswith('F') else float(user_in[:-1])
        engine = ThermalEngine(i_raw, ambient)
        engine.ground_alt_m = 44330 * (1.0 - math.pow(i_press / P_STANDARD, 0.1903))

        def data_callback(timestamp, data, logconf):
            raw_t, press, thrust, vbat = data['baro.temp'], data['baro.pressure'], data['stabilizer.thrust'], data['pm.vbat']
            corr_t, roc, abs_alt, dens, qff = engine.process(raw_t, thrust, press)
            rel_alt = abs_alt - engine.ground_alt_m

            print(f"\n{'='*25} System Status [{timestamp}ms] {'='*25}")
            print(f"STATUS:  {engine.state} | {engine.last_event}")
            print(f"ROC:     {roc:.4f}°C/s | Trend Samples: {engine.trend_counter}")
            
            print(f"\nTEMPERATURE:")
            print(f"  Raw:       {raw_t:.2f}°C / {c_to_f(raw_t):.2f}°F")
            print(f"  Corrected: {corr_t:.2f}°C / {c_to_f(corr_t):.2f}°F")
            print(f"  Offset:    {engine.idle_offset:.4f}")

            print(f"\nALTITUDE & AIR:")
            print(f"  Relative:  {rel_alt:.2f} m / {m_to_ft(rel_alt):.2f} ft (AGL)")
            print(f"  Density:   {dens:.4f} kg/m³")
            
            print(f"\nPRESSURE & POWER:")
            print(f"  Station P: {press:.2f} hPa | Sea Level (QFF): {qff:.2f} hPa")
            print(f"  Battery:   {vbat:.2f} V   | Thrust: {thrust}")
            print(f"{'='*65}")
            print("Keys: 't' = Temp Reset | 'z' = Zero Altitude | 'Ctrl+C' = Quit")

        full_log = LogConfig(name='Master', period_in_ms=500)
        vars = [('baro.temp','float'),('baro.pressure','float'),('pm.vbat','float'),('stabilizer.thrust','uint16_t')]
        for v, t in vars: full_log.add_variable(v, t)

        scf.cf.log.add_config(full_log); full_log.data_received_cb.add_callback(data_callback); full_log.start()

        try:
            print("Control loop active. Press 't' to reset temp, 'z' to zero altitude.")
            while True:
                # Check if a key has been pressed
                if msvcrt.kbhit():
                    # Read the key (decode from bytes to string)
                    key = msvcrt.getch().decode('utf-8').lower()
                    
                    if key == 't':
                        engine.idle_offset = engine.ema_temp - engine.ambient_baseline
                        print("\n>>> Temperature Offset Reset!")
                    elif key == 'z':
                        # Note: 'press' needs to be accessible here; 
                        # using the last known pressure from the engine logic
                        # is usually the safest bet in this scope.
                        engine.ground_alt_m = 44330 * (1.0 - math.pow(engine.prev_press / P_STANDARD, 0.1903))
                        print("\n>>> Altitude Zeroed!")
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            full_log.stop()

if __name__ == '__main__':
    run_master_engine()