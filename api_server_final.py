from __future__ import annotations

import csv
import json
import os
import queue
import re
import socketserver
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / 'mainFrontEnd_final.html'
CSV_FILE = BASE_DIR / 'drone_mapping_data.csv'
CSV_ROOT = BASE_DIR / 'organized_flights'
MISSION_PATHS_DIR = BASE_DIR / 'mission_paths'
HOST = '127.0.0.1'
PORT = 5001

# --- LLM (optional): set CW_LLM_API_KEY_INLINE to skip env vars; do not commit secrets ---
CW_LLM_API_KEY_INLINE = ''  # e.g. 'sk-...'
CW_LLM_BASE_URL_INLINE = ''  # empty = https://api.openai.com/v1
CW_LLM_MODEL_INLINE = ''  # empty = gpt-4o-mini

LLM_API_KEY = (
    os.environ.get('OPENAI_API_KEY')
    or os.environ.get('CW_LLM_API_KEY')
    or (CW_LLM_API_KEY_INLINE.strip() if CW_LLM_API_KEY_INLINE else '')
)
LLM_BASE_URL = (
    os.environ.get('OPENAI_BASE_URL')
    or (CW_LLM_BASE_URL_INLINE.strip() if CW_LLM_BASE_URL_INLINE else '')
    or 'https://api.openai.com/v1'
).rstrip('/')
LLM_MODEL = (
    os.environ.get('CW_LLM_MODEL')
    or (CW_LLM_MODEL_INLINE.strip() if CW_LLM_MODEL_INLINE else '')
    or 'gpt-4o-mini'
)


class TelemetryState:
    def __init__(self):
        self.lock = Lock()
        self.connected = False
        self.last_seen = 0.0
        self.telemetry = {}
        self.geofence = {'x_min': 0.0, 'x_max': 3.0, 'y_min': 0.0, 'y_max': 3.0, 'violated': False}
        self.mission = {'flight_count': 0}
        self.subscribers: list[queue.Queue] = []

        # --- Mission bridge state ---
        # A single-slot inbox for the ROS bridge to poll.
        # None means "no new mission"; otherwise a list of {x, y, z} dicts.
        self.pending_mission: list | None = None
        # Monotonically increasing counter. Incremented on every cancel.
        # Bridge pulls it on every poll; any mid-execution increment tells
        # the bridge to abort what it is currently running.
        self.cancel_epoch: int = 0
        # Latest status reported by the bridge (for GET /api/mission/status).
        self.mission_status = {
            'state': 'idle',       # idle | queued | running | done | cancelled | error
            'current_idx': 0,
            'total': 0,
            'message': '',
            'epoch': 0,
        }
        # Liveness probe for the bridge: last time it hit us with anything.
        self.bridge_last_seen: float = 0.0

    def snapshot(self):
        with self.lock:
            connected = self.connected and (time.time() - self.last_seen) < 5.0
            bridge_connected = (time.time() - self.bridge_last_seen) < 5.0
            return {
                'connected': connected,
                'telemetry': dict(self.telemetry),
                'geofence': dict(self.geofence),
                'mission': dict(self.mission),
                'mission_status': dict(self.mission_status),
                'bridge_connected': bridge_connected,
                'cancel_epoch': self.cancel_epoch,
            }

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def broadcast(self, event_type, payload):
        with self.lock:
            subscribers = list(self.subscribers)
        dead = []
        for q in subscribers:
            try:
                q.put((event_type, payload), timeout=0.1)
            except Exception:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)


STATE = TelemetryState()


def json_bytes(obj):
    return json.dumps(obj).encode('utf-8')


def coerce(value):
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s == '':
        return s
    for cast in (int, float):
        try:
            return cast(s)
        except Exception:
            pass
    return s


def _ensure_mission_dir():
    MISSION_PATHS_DIR.mkdir(parents=True, exist_ok=True)


def zone_center_xy(label: str, gh_rows: int, gh_cols: int, gf: dict) -> tuple[float, float] | None:
    """Match frontend: cell center from zone label (e.g. A1) and geofence."""
    label = (label or '').strip().upper()
    if len(label) < 2:
        return None
    row = ord(label[0]) - ord('A')
    try:
        col = int(label[1:]) - 1
    except ValueError:
        return None
    if not (0 <= row < gh_rows and 0 <= col < gh_cols):
        return None
    spacing_x = (float(gf['x_max']) - float(gf['x_min'])) / float(gh_cols)
    spacing_y = (float(gf['y_max']) - float(gf['y_min'])) / float(gh_rows)
    x = float(gf['x_min']) + (col + 0.5) * spacing_x
    y = float(gf['y_min']) + (row + 0.5) * spacing_y
    return x, y


def build_mission_txt(waypoints: list[dict], gh_rows: int, gh_cols: int, gf: dict) -> str:
    """Human- and parser-friendly text; same semantic columns as CSV export."""
    lines = [
        '# CropWatcher flight path v1',
        f'# grid: {gh_rows}x{gh_cols}  geofence x:[{gf["x_min"]},{gf["x_max"]}] y:[{gf["y_min"]},{gf["y_max"]}] (meters)',
        '# Data lines: seq zone_id x_m y_m z_m',
    ]
    for i, wp in enumerate(waypoints, start=1):
        lab = str(wp.get('label') or wp.get('zone_id') or '?')
        x = float(wp['x'])
        y = float(wp['y'])
        z = float(wp.get('z', 0.5))
        lines.append(f'{i} {lab} {x:.6f} {y:.6f} {z:.4f}')
    return '\n'.join(lines) + '\n'


def save_mission_txt_file(content: str) -> tuple[Path, Path]:
    """Writes timestamped file and overwrites latest_mission.txt for backend pickup."""
    _ensure_mission_dir()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    stamped = MISSION_PATHS_DIR / f'mission_{ts}.txt'
    latest = MISSION_PATHS_DIR / 'latest_mission.txt'
    stamped.write_text(content, encoding='utf-8')
    latest.write_text(content, encoding='utf-8')
    return stamped, latest


def extract_json_object(text: str) -> dict:
    """Parse first JSON object from model output (strips optional markdown fences)."""
    t = text.strip()
    if '```' in t:
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', t, re.IGNORECASE)
        if m:
            t = m.group(1).strip()
    if not t.startswith('{'):
        i = t.find('{')
        if i >= 0:
            t = t[i:]
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(t)
    if not isinstance(obj, dict):
        raise ValueError('LLM output must be a JSON object')
    return obj


def llm_chat_completion(system: str, user: str) -> str:
    if not LLM_API_KEY:
        raise RuntimeError('LLM API key not configured (set OPENAI_API_KEY or CW_LLM_API_KEY)')
    url = f'{LLM_BASE_URL}/chat/completions'
    body = json.dumps({
        'model': LLM_MODEL,
        'temperature': 0.2,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
    }).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {LLM_API_KEY}',
        },
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = json.loads(resp.read().decode('utf-8'))
    choices = raw.get('choices') or []
    if not choices:
        raise RuntimeError('LLM returned no choices')
    return (choices[0].get('message') or {}).get('content') or ''


def _list_date_dirs():
    if not CSV_ROOT.exists():
        return []
    dates = [p.name for p in CSV_ROOT.iterdir() if p.is_dir()]
    # ISO date folder names sort correctly as strings
    return sorted(dates, reverse=True)


def _csv_files_for_date(date_str: str):
    date_dir = CSV_ROOT / date_str
    if not date_dir.exists() or not date_dir.is_dir():
        return []
    return sorted([p for p in date_dir.glob('*.csv') if p.is_file()], reverse=True)


def _resolve_csv_path(date_str: str | None = None, file_name: str | None = None):
    # Priority 1: explicit date folder from organized output
    if date_str:
        files = _csv_files_for_date(date_str)
        if file_name:
            for f in files:
                if f.name == file_name:
                    return f
            return None
        if files:
            return files[0]
        return None

    # Priority 2: latest dated folder/file in organized output
    for d in _list_date_dirs():
        files = _csv_files_for_date(d)
        if files:
            return files[0]

    # Priority 3: legacy single CSV file
    return CSV_FILE if CSV_FILE.exists() else None


# ============================================================
#                  FLIGHT SCRIPT LAUNCHER
# ============================================================
PYTHON_EXE = sys.executable
SCRIPT_FILES = {
    'waypoint': ['waypoint_flight.py', 'waypoint_flight_feet_teamlogic.py'],
    'hover': ['hover_test.py'],
    'lawnmower': ['lawnmower_flight.py'],
}
flight_process: subprocess.Popen | None = None
flight_process_name: str | None = None
flight_output_lines: list[str] = []
flight_lock = threading.Lock()
MAX_FLIGHT_OUTPUT_LINES = 800


def _flight_add_output(line: str) -> None:
    with flight_lock:
        flight_output_lines.append(line.rstrip('\n'))
        if len(flight_output_lines) > MAX_FLIGHT_OUTPUT_LINES:
            del flight_output_lines[: len(flight_output_lines) - MAX_FLIGHT_OUTPUT_LINES]


def _flight_get_output() -> str:
    with flight_lock:
        return '\n'.join(flight_output_lines[-MAX_FLIGHT_OUTPUT_LINES:])


def _flight_find_script(script_key: str) -> Path | None:
    for name in SCRIPT_FILES.get(script_key, []):
        candidate = BASE_DIR / name
        if candidate.exists():
            return candidate
    return None


def _flight_process_is_running() -> bool:
    global flight_process
    return flight_process is not None and flight_process.poll() is None


def _flight_reader_thread(proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        _flight_add_output(line)
    rc = proc.wait()
    _flight_add_output(f"\n[telemetry_api_server] Flight process exited with code {rc}")
    STATE.broadcast('event', {'level': 'info', 'message': f'Flight process exited with code {rc}'})


def launch_flight_script(payload: dict[str, Any]) -> dict[str, Any]:
    global flight_process, flight_process_name, flight_output_lines

    if _flight_process_is_running():
        return {'ok': False, 'error': f'A flight script is already running: {flight_process_name}'}

    script_key = str(payload.get('script', '')).strip().lower()
    script_path = _flight_find_script(script_key)
    if script_path is None:
        expected = ', '.join(SCRIPT_FILES.get(script_key, [])) or 'waypoint/hover/lawnmower'
        return {'ok': False, 'error': f"Could not find script for '{script_key}'. Expected: {expected} in {BASE_DIR}"}

    uri = str(payload.get('uri', 'radio://0/80/2M/E7E7E7E7E7')).strip()
    hover_altitude_ft = str(payload.get('hover_altitude_ft', payload.get('altitude_ft', '2.0'))).strip()
    hover_duration_s = str(payload.get('hover_duration_s', payload.get('duration_s', '10'))).strip()
    ambient_temp = str(payload.get('ambient_temp', '74F')).strip()

    cmd = [PYTHON_EXE, '-u', str(script_path)]
    stdin_text = None

    if script_key == 'waypoint':
        cmd += ['--uri', uri]
        # waypoint_flight.py now reuses lawnmower_flight.py SensorLogger, which prompts for ambient temp
        # once the real flight begins. Keep stdin open long enough for that prompt.
        stdin_text = ambient_temp + '\n'
    elif script_key == 'hover':
        # hover_test.py prompts for altitude in feet; some versions also prompt for duration.
        stdin_text = hover_altitude_ft + '\n' + hover_duration_s + '\n'
    elif script_key == 'lawnmower':
        # lawnmower_flight.py prompts for ambient room temperature.
        stdin_text = ambient_temp + '\n'

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['CFLIB_URI'] = uri
    env['CRAZYFLIE_URI'] = uri

    with flight_lock:
        flight_output_lines = [
            f'[telemetry_api_server] Launching {script_path.name}',
            f'[telemetry_api_server] Working directory: {BASE_DIR}',
            "[telemetry_api_server] Command: " + " ".join(cmd),
        ]

    flight_process_name = script_path.name
    flight_process = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        stdin=subprocess.PIPE if stdin_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if stdin_text is not None and flight_process.stdin is not None:
        try:
            flight_process.stdin.write(stdin_text)
            flight_process.stdin.flush()
            # Do not close stdin for waypoint because Tk stays open and SensorLogger may prompt after button click.
            if script_key != 'waypoint':
                flight_process.stdin.close()
        except BrokenPipeError:
            pass

    threading.Thread(target=_flight_reader_thread, args=(flight_process,), daemon=True).start()
    STATE.broadcast('event', {'level': 'info', 'message': f'Started {script_path.name}'})

    return {
        'ok': True,
        'message': f'Started {script_path.name}',
        'script_file': script_path.name,
        'pid': flight_process.pid,
    }


def stop_flight_script() -> dict[str, Any]:
    global flight_process
    if not _flight_process_is_running():
        return {'ok': True, 'message': 'No flight script is currently running.', 'output': _flight_get_output()}

    assert flight_process is not None
    _flight_add_output('[telemetry_api_server] Stop requested by front end.')
    try:
        if os.name == 'nt':
            flight_process.terminate()
        else:
            flight_process.send_signal(signal.SIGINT)
        try:
            flight_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _flight_add_output('[telemetry_api_server] Process did not stop after SIGINT; terminating.')
            flight_process.terminate()
            try:
                flight_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _flight_add_output('[telemetry_api_server] Process did not terminate; killing.')
                flight_process.kill()
    finally:
        STATE.broadcast('event', {'level': 'warn', 'message': 'Flight stop signal sent'})
        return {'ok': True, 'message': 'Stop signal sent.', 'output': _flight_get_output()}


def flight_status_payload() -> dict[str, Any]:
    return {
        'ok': True,
        'running': _flight_process_is_running(),
        'script': flight_process_name,
        'pid': flight_process.pid if flight_process is not None and _flight_process_is_running() else None,
        'output': _flight_get_output(),
        'scripts_found': {key: str(_flight_find_script(key).name) if _flight_find_script(key) else None for key in SCRIPT_FILES},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = 'TelemetryHTTP/1.0'

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == '/':
            return self.serve_file(HTML_FILE, 'text/html; charset=utf-8')
        if path == '/api/status':
            return self.send_json(STATE.snapshot())
        if path == '/api/flight/status':
            return self.send_json(flight_status_payload())
        if path == '/api/health':
            resolved = _resolve_csv_path()
            _ensure_mission_dir()
            latest_m = MISSION_PATHS_DIR / 'latest_mission.txt'
            return self.send_json({
                'ok': True,
                'html': HTML_FILE.name,
                'csv_exists': resolved is not None,
                'csv_source': str(resolved) if resolved else None,
                'available_dates': _list_date_dirs(),
                'mission_paths_dir': str(MISSION_PATHS_DIR),
                'latest_mission_txt': str(latest_m) if latest_m.exists() else None,
                'llm_configured': bool(LLM_API_KEY),
            })
        if path == '/api/csv/dates':
            return self.handle_csv_dates()
        if path == '/api/csv/files':
            return self.handle_csv_files(parsed)
        if path == '/api/csv/rows':
            return self.handle_csv_rows(parsed)
        if path == '/api/stream':
            return self.handle_stream()
        if path == '/api/mission/pop':
            # ROS bridge polls this. Returns {'mission': [...], 'cancel_epoch': N}
            # or {'mission': None, 'cancel_epoch': N}. Bridge updates its own
            # view of cancel_epoch every poll and aborts in-flight work if it
            # sees the number jump.
            with STATE.lock:
                STATE.bridge_last_seen = time.time()
                mission = STATE.pending_mission
                STATE.pending_mission = None
                epoch = STATE.cancel_epoch
                if mission is not None:
                    STATE.mission_status = {
                        'state': 'running',
                        'current_idx': 0,
                        'total': len(mission),
                        'message': 'bridge picked up mission',
                        'epoch': epoch,
                    }
            if mission is not None:
                STATE.broadcast('event', {
                    'level': 'info',
                    'message': f'Bridge picked up mission ({len(mission)} waypoints)',
                })
            return self.send_json({'mission': mission, 'cancel_epoch': epoch})
        if path == '/api/mission/status':
            with STATE.lock:
                bridge_alive = (time.time() - STATE.bridge_last_seen) < 5.0
                payload = {
                    **STATE.mission_status,
                    'bridge_connected': bridge_alive,
                    'cancel_epoch': STATE.cancel_epoch,
                    'pending_count': len(STATE.pending_mission) if STATE.pending_mission else 0,
                }
            return self.send_json(payload)
        self.send_error(HTTPStatus.NOT_FOUND, 'Not found')

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == '/api/telemetry':
            payload = self.read_json_body()
            with STATE.lock:
                STATE.telemetry = payload
                STATE.connected = True
                STATE.last_seen = time.time()
            STATE.broadcast('telemetry', payload)
            return self.send_json({'ok': True})
        if path == '/api/event':
            payload = self.read_json_body()
            STATE.broadcast('event', payload)
            return self.send_json({'ok': True})
        if path == '/api/flight/run':
            payload = self.read_json_body()
            result = launch_flight_script(payload)
            return self.send_json(result, status=HTTPStatus.OK if result.get('ok') else HTTPStatus.BAD_REQUEST)
        if path == '/api/flight/stop':
            return self.send_json(stop_flight_script())
        if path == '/api/hover-test':
            payload = self.read_json_body()
            payload['script'] = 'hover'
            result = launch_flight_script(payload)
            return self.send_json(result, status=HTTPStatus.OK if result.get('ok') else HTTPStatus.BAD_REQUEST)
        if path == '/api/mission':
            # Frontend "Execute Mission" hits this.
            # Body: {"waypoints": [{x,y,z,label?}, ...], "gh_rows"?, "gh_cols"?}
            payload = self.read_json_body()
            wps_raw = payload.get('waypoints') or []
            if not isinstance(wps_raw, list) or not wps_raw:
                return self.send_json(
                    {'ok': False, 'error': 'waypoints must be a non-empty list'},
                    status=HTTPStatus.BAD_REQUEST,
                )
            gh_rows = int(payload.get('gh_rows') or 3)
            gh_cols = int(payload.get('gh_cols') or 4)
            # Normalize + geofence-check on the server side so an offline
            # bridge can't swallow bad points silently.
            cleaned = []
            with STATE.lock:
                gf = dict(STATE.geofence)
            for i, wp in enumerate(wps_raw):
                try:
                    x = float(wp['x']); y = float(wp['y'])
                    z = float(wp.get('z', 0.5))
                except (KeyError, TypeError, ValueError):
                    return self.send_json(
                        {'ok': False, 'error': f'waypoint[{i}] missing x/y or not numeric'},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                if not (gf['x_min'] <= x <= gf['x_max'] and gf['y_min'] <= y <= gf['y_max']):
                    return self.send_json(
                        {'ok': False, 'error': f'waypoint[{i}] ({x}, {y}) outside geofence'},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                lab = wp.get('label') or wp.get('zone_id') or f'WP{i + 1}'
                cleaned.append({'x': x, 'y': y, 'z': z, 'label': str(lab)})
            txt_body = build_mission_txt(cleaned, gh_rows, gh_cols, gf)
            stamped, latest = save_mission_txt_file(txt_body)
            with STATE.lock:
                STATE.pending_mission = [{'x': w['x'], 'y': w['y'], 'z': w['z']} for w in cleaned]
                STATE.mission_status = {
                    'state': 'queued',
                    'current_idx': 0,
                    'total': len(cleaned),
                    'message': 'waiting for bridge to poll',
                    'epoch': STATE.cancel_epoch,
                }
                bridge_alive = (time.time() - STATE.bridge_last_seen) < 5.0
            STATE.broadcast('event', {
                'level': 'info' if bridge_alive else 'warn',
                'message': (f'Mission queued: {len(cleaned)} waypoints'
                            if bridge_alive
                            else f'Mission queued ({len(cleaned)} waypoints) '
                                 f'but bridge is not connected'),
            })
            return self.send_json({
                'ok': True,
                'queued': len(cleaned),
                'bridge_connected': bridge_alive,
                'saved_txt': str(stamped.relative_to(BASE_DIR)),
                'latest_txt': str(latest.relative_to(BASE_DIR)),
                'mission_txt_preview': txt_body[:2000],
            })
        if path == '/api/mission/chat':
            # Natural language -> same mission txt + optional queue.
            # Body: { "message": str, "context": { gh_rows, gh_cols, geofence?, default_z? }, "queue_mission": bool }
            payload = self.read_json_body()
            msg = (payload.get('message') or '').strip()
            if not msg:
                return self.send_json({'ok': False, 'error': 'message is required'}, status=HTTPStatus.BAD_REQUEST)
            ctx = payload.get('context') or {}
            gh_rows = int(ctx.get('gh_rows') or 3)
            gh_cols = int(ctx.get('gh_cols') or 4)
            default_z = float(ctx.get('default_z') or 0.6)
            with STATE.lock:
                gf = dict(STATE.geofence)
            if isinstance(ctx.get('geofence'), dict):
                g = ctx['geofence']
                for k in ('x_min', 'x_max', 'y_min', 'y_max'):
                    if k in g:
                        gf[k] = float(g[k])

            zone_examples = []
            for r in range(min(gh_rows, 3)):
                for c in range(min(gh_cols, 4)):
                    lab = f'{chr(65 + r)}{c + 1}'
                    zone_examples.append(lab)

            system = (
                'You are a greenhouse drone path planner. '
                f'The grid has {gh_rows} rows (letters A+) and {gh_cols} columns (numbers 1+). '
                f'Valid zone labels look like: {", ".join(zone_examples)}. '
                'Reply with ONE JSON object only, no markdown outside JSON. Schema:\n'
                '{"sequence":["A1","B2",...],"z":0.6}\n'
                'Use "sequence" as visit order (repeat labels allowed if user asks). '
                'Optional per-point z: {"sequence":[{"label":"A1","z":0.5},{"label":"B2"}]} — if you use objects, each item must have "label". '
                'If only heights differ, you may use uniform "z" at top level.'
            )
            raw = ''
            try:
                raw = llm_chat_completion(system, msg)
                data = extract_json_object(raw)
            except ValueError as e:
                return self.send_json(
                    {'ok': False, 'error': str(e), 'raw': raw[:4000]},
                    status=HTTPStatus.BAD_REQUEST,
                )
            except json.JSONDecodeError as e:
                return self.send_json(
                    {'ok': False, 'error': f'LLM JSON parse error: {e}', 'raw': raw[:4000]},
                    status=HTTPStatus.BAD_REQUEST,
                )
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read().decode('utf-8', errors='replace')[:500]
                except Exception:
                    detail = str(e)
                return self.send_json(
                    {'ok': False, 'error': f'LLM HTTP {e.code}: {detail}'},
                    status=HTTPStatus.BAD_REQUEST,
                )
            except Exception as e:
                return self.send_json(
                    {'ok': False, 'error': str(e)},
                    status=HTTPStatus.BAD_REQUEST,
                )

            seq = data.get('sequence') or data.get('waypoints') or data.get('path')
            if not isinstance(seq, list) or not seq:
                return self.send_json(
                    {'ok': False, 'error': 'LLM JSON must include non-empty "sequence" array'},
                    status=HTTPStatus.BAD_REQUEST,
                )
            z_uniform = data.get('z', default_z)
            cleaned = []
            for i, item in enumerate(seq):
                if isinstance(item, dict):
                    lab = item.get('label') or item.get('zone') or item.get('id')
                    z = float(item.get('z', z_uniform))
                else:
                    lab = str(item).strip()
                    z = float(z_uniform)
                if not lab:
                    continue
                xy = zone_center_xy(str(lab), gh_rows, gh_cols, gf)
                if xy is None:
                    return self.send_json(
                        {'ok': False, 'error': f'Invalid or out-of-grid zone: {lab}'},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                x, y = xy
                if not (gf['x_min'] <= x <= gf['x_max'] and gf['y_min'] <= y <= gf['y_max']):
                    return self.send_json(
                        {'ok': False, 'error': f'Zone {lab} maps outside geofence'},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                canon = str(lab).strip().upper()
                cleaned.append({'x': x, 'y': y, 'z': z, 'label': canon})

            if not cleaned:
                return self.send_json({'ok': False, 'error': 'No valid waypoints after parsing'}, status=HTTPStatus.BAD_REQUEST)

            txt_body = build_mission_txt(cleaned, gh_rows, gh_cols, gf)
            stamped, latest = save_mission_txt_file(txt_body)
            queue_mission = bool(payload.get('queue_mission'))
            bridge_alive = False
            if queue_mission:
                with STATE.lock:
                    STATE.pending_mission = [{'x': w['x'], 'y': w['y'], 'z': w['z']} for w in cleaned]
                    STATE.mission_status = {
                        'state': 'queued',
                        'current_idx': 0,
                        'total': len(cleaned),
                        'message': 'waiting for bridge to poll (from LLM)',
                        'epoch': STATE.cancel_epoch,
                    }
                    bridge_alive = (time.time() - STATE.bridge_last_seen) < 5.0
                STATE.broadcast('event', {
                    'level': 'info' if bridge_alive else 'warn',
                    'message': f'LLM mission queued: {len(cleaned)} waypoints',
                })

            summary = (
                f'Generated {len(cleaned)} waypoints; file written to {latest.name}.'
                + (' Mission queued.' if queue_mission else ' Not queued — click Execute Mission in the UI if needed.')
            )
            return self.send_json({
                'ok': True,
                'waypoints': cleaned,
                'saved_txt': str(stamped.relative_to(BASE_DIR)),
                'latest_txt': str(latest.relative_to(BASE_DIR)),
                'mission_txt_preview': txt_body[:2000],
                'summary': summary,
                'queued': len(cleaned) if queue_mission else 0,
                'bridge_connected': bridge_alive,
            })

        if path == '/api/mission/cancel':
            # Emergency cancel. Bumps cancel_epoch and clears any pending
            # mission. The bridge sees the epoch jump on its next poll and
            # aborts whatever it is doing.
            with STATE.lock:
                STATE.cancel_epoch += 1
                STATE.pending_mission = None
                STATE.mission_status = {
                    'state': 'cancelled',
                    'current_idx': STATE.mission_status.get('current_idx', 0),
                    'total': STATE.mission_status.get('total', 0),
                    'message': 'cancelled by user',
                    'epoch': STATE.cancel_epoch,
                }
                new_epoch = STATE.cancel_epoch
            STATE.broadcast('event', {
                'level': 'warn',
                'message': f'Mission cancel requested (epoch={new_epoch})',
            })
            return self.send_json({'ok': True, 'cancel_epoch': new_epoch})
        if path == '/api/mission/progress':
            # Bridge reports progress. Body: {"state": "...", "current_idx": N,
            #   "total": N, "message": "..."}
            payload = self.read_json_body()
            with STATE.lock:
                STATE.bridge_last_seen = time.time()
                STATE.mission_status = {
                    'state': payload.get('state', STATE.mission_status.get('state', 'running')),
                    'current_idx': int(payload.get('current_idx', 0)),
                    'total': int(payload.get('total', 0)),
                    'message': str(payload.get('message', '')),
                    'epoch': STATE.cancel_epoch,
                }
            STATE.broadcast('mission', {**payload, 'epoch': STATE.cancel_epoch})
            return self.send_json({'ok': True})
        self.send_error(HTTPStatus.NOT_FOUND, 'Not found')

    def serve_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, 'Missing file')
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, obj, status=HTTPStatus.OK):
        data = json_bytes(obj)
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json_body(self):
        length = int(self.headers.get('Content-Length', '0') or 0)
        raw = self.rfile.read(length) if length else b'{}'
        try:
            return json.loads(raw.decode('utf-8'))
        except Exception:
            return {}

    def handle_csv_dates(self):
        dates = _list_date_dirs()
        return self.send_json({'dates': dates, 'latest_date': (dates[0] if dates else None)})

    def handle_csv_files(self, parsed):
        qs = parse_qs(parsed.query or '')
        date_str = (qs.get('date') or [None])[0]
        if not date_str:
            return self.send_json({'files': []})
        files = _csv_files_for_date(date_str)
        return self.send_json({'files': [f.name for f in files], 'date': date_str})

    def handle_csv_rows(self, parsed):
        qs = parse_qs(parsed.query or '')
        date_str = (qs.get('date') or [None])[0]
        file_name = (qs.get('file') or [None])[0]
        csv_path = _resolve_csv_path(date_str, file_name=file_name)
        if csv_path is None:
            return self.send_json({'rows': [], 'columns': [], 'date': date_str, 'file': file_name, 'source': None})
        rows = []
        with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            for row in reader:
                rows.append({k: coerce(v) for k, v in row.items()})
        resolved_date = csv_path.parent.name if csv_path.parent.parent == CSV_ROOT else None
        return self.send_json({
            'rows': rows,
            'columns': columns,
            'date': resolved_date,
            'file': csv_path.name,
            'source': str(csv_path),
        })

    def handle_stream(self):
        q = STATE.subscribe()
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()
        try:
            self.write_sse('init', STATE.snapshot())
            while True:
                try:
                    event_type, payload = q.get(timeout=15)
                    self.write_sse(event_type, payload)
                except queue.Empty:
                    self.wfile.write(b': keep-alive\n\n')
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            STATE.unsubscribe(q)

    def write_sse(self, event_type, payload):
        frame = f'event: {event_type}\ndata: {json.dumps(payload)}\n\n'.encode('utf-8')
        self.wfile.write(frame)
        self.wfile.flush()

    def log_message(self, format, *args):
        return


if __name__ == '__main__':
    _ensure_mission_dir()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f'Serving frontend: {HTML_FILE}')
    print(f'CSV source (legacy): {CSV_FILE}')
    print(f'CSV root (organized): {CSV_ROOT}')
    print(f'Mission path files: {MISSION_PATHS_DIR} (latest: latest_mission.txt)')
    print(f'Open: http://{HOST}:{PORT}')
    print('Flight launcher endpoints are included on this same server: /api/flight/status, /api/flight/run, /api/flight/stop')
    server.serve_forever()