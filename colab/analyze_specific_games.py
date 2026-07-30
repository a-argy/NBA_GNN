#!/usr/bin/env python3
"""
Download two specific NBA games, run shot extraction, and produce a report of:
  - Every captured shot (number, player, timestamp, result)
  - Every dropped shot with its reason and extra detail in one unified table
"""
import sys
import os
import json
import requests
import py7zr
import pandas as pd

COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, COLAB_DIR)
import load_with_context as lwc

# ── Config ───────────────────────────────────────────────────────────────────
GAMES = [
    ("12.25.2015.NOP.at.MIA.7z", "New Orleans Pelicans @ Miami Heat — December 25, 2015"),
    ("12.28.2015.SAC.at.GSW.7z", "Sacramento Kings @ Golden State Warriors — December 28, 2015"),
]

BASE_URL = (
    "https://raw.githubusercontent.com/linouk23/NBA-Player-Movements"
    "/master/data/2016.NBA.Raw.SportVU.Game.Logs/"
)
PBP_URL = (
    "https://github.com/sumitrodatta/nba-alt-awards/raw/main"
    "/Historical/PBP%20Data/2015-16_pbp.csv"
)

DATA_DIR = os.path.join(COLAB_DIR, "data", "game_data")
PBP_PATH = os.path.join(COLAB_DIR, "data", "supplemental_data", "pbp_cache.csv")
OUTPUT   = os.path.join(COLAB_DIR, "data", "shot_report.txt")


# ── Download / extract ────────────────────────────────────────────────────────
def download_and_extract(filename):
    os.makedirs(DATA_DIR, exist_ok=True)
    seven_z = os.path.join(DATA_DIR, filename)
    url = BASE_URL + filename

    if not os.path.exists(seven_z):
        print(f"  Downloading {filename} ...")
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} — {url}")
        if not r.content.startswith(b"7z\xbc\xaf\x27\x1c"):
            raise RuntimeError("Response is not a valid 7z archive")
        with open(seven_z, "wb") as f:
            f.write(r.content)

    with py7zr.SevenZipFile(seven_z, mode="r") as arc:
        names = arc.getnames()
    json_names = [n for n in names if n.endswith(".json")]
    if not json_names:
        raise RuntimeError(f"No .json inside {filename}")
    json_name = json_names[0]
    json_path = os.path.join(DATA_DIR, json_name)

    if os.path.exists(json_path):
        print(f"  Cached: {json_name}")
        os.remove(seven_z)
        return json_path

    print(f"  Extracting → {json_name} ...")
    with py7zr.SevenZipFile(seven_z, mode="r") as arc:
        arc.extractall(path=DATA_DIR)
    os.remove(seven_z)
    print(f"  Done")
    return json_path


def get_pbp():
    if os.path.exists(PBP_PATH):
        print(f"PBP: loaded from cache")
        return pd.read_csv(PBP_PATH)
    print("PBP: downloading ...")
    r = requests.get(PBP_URL, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} downloading PBP")
    os.makedirs(os.path.dirname(PBP_PATH), exist_ok=True)
    with open(PBP_PATH, "wb") as f:
        f.write(r.content)
    print(f"PBP: saved to {PBP_PATH}")
    return pd.read_csv(PBP_PATH)


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_pbp_time(time_str):
    try:
        parts = str(time_str).strip().split(":")
        return int(parts[0]) * 60 + float(parts[1])
    except Exception:
        return 0.0


def fmt_tracking(seconds, quarter):
    m, s = divmod(int(seconds), 60)
    return f"Q{quarter} {m}:{s:02d}"


def pbp_desc(row):
    return (
        str(row.get("HOMEDESCRIPTION") or "")
        + str(row.get("VISITORDESCRIPTION") or "")
        + str(row.get("NEUTRALDESCRIPTION") or "")
    ).strip()


def restructure_moments(event):
    """Convert raw moment arrays to named-field dicts (same as load_shot_attempts)."""
    moments = []
    for m in event.get("moments", []):
        try:
            moments.append({
                "quarter": m[0],
                "game_clock": m[2],
                "shot_clock": m[3],
                "ball_coordinates": {"x": m[5][0][2], "y": m[5][0][3], "z": m[5][0][4]},
                "player_coordinates": [
                    {"teamid": p[0], "playerid": p[1], "x": p[2], "y": p[3], "z": p[4]}
                    for p in m[5][1:]
                ],
            })
        except (IndexError, TypeError):
            pass
    return moments


# ── Per-shot outcome tracing ──────────────────────────────────────────────────
DROP_REASON_LABELS = {
    "not_enough_moments":           "Fewer than 2 moments in event",
    "no_rim_moment_found":          "Ball never approached rim",
    "no_player_within_threshold":   "No player near ball at release",
    "first_player_not_shooter":     "Nearest player was not the shooter",
    "not_enough_history":           "Insufficient historical moments",
    "rescued_by_secondary_event":   "Rescued by secondary event fallback",
    "time_proximity_failed":        "Release time too far from PBP time",
}


def trace_all_shot_outcomes(game, pbp, pname):
    """
    Walk every PBP shot event through the same decision tree as load_shot_attempts
    and record the outcome (extracted or dropped + reason + detail) for each.

    NOTE: calibrate_rim_assignments must have already been called for this game
    (load_shot_attempts does this) so the module-level rim globals are set.
    """
    game_id = game["gameid"]

    def get_name(pid):
        try:
            return pname.get(int(pid), f"Player #{pid}")
        except Exception:
            return f"Player #{pid}"

    game_pbp    = pbp[pbp.GAME_ID == int(game_id)]
    shot_events = game_pbp[game_pbp.EVENTMSGTYPE.isin([1, 2])]

    results = []

    for _, row in shot_events.iterrows():
        poss_team_id  = row["PLAYER1_TEAM_ID"]
        shooter_id_raw = row["PLAYER1_ID"]
        quarter  = int(row["PERIOD"])
        time_str = str(row["PCTIMESTRING"])
        made     = int(row["EVENTMSGTYPE"]) == 1
        desc     = pbp_desc(row)

        entry = {
            "quarter":      quarter,
            "time":         time_str,
            "shooter_id":   None,
            "shooter_name": "Unknown",
            "made":         made,
            "description":  desc,
            "status":       "dropped",
            "drop_reason":  None,
            "drop_detail":  None,
        }

        # Stage 1: PBP has team + shooter?
        if pd.isna(poss_team_id) or pd.isna(shooter_id_raw):
            entry["drop_reason"] = "Missing team/shooter ID in PBP"
            results.append(entry)
            continue

        poss_team_id = int(poss_team_id)
        shooter_id   = int(shooter_id_raw)
        entry["shooter_id"]   = shooter_id
        entry["shooter_name"] = get_name(shooter_id)

        # Stage 2: Find all candidate tracking events (sorted best-first)
        pbp_secs = parse_pbp_time(time_str)
        candidates = lwc.find_candidate_events(game["events"], quarter, pbp_secs)
        if not candidates:
            entry["drop_reason"] = "No matching tracking event (time tolerance)"
            results.append(entry)
            continue

        # Stage 3 & 4: Try each candidate until one yields a valid release moment
        release_moments = None
        drop_reason = None
        debug_info = None

        for candidate_event in candidates:
            moments = restructure_moments(candidate_event)
            if not moments:
                continue

            actual_quarter = moments[0]["quarter"]
            target_rim = lwc.determine_target_rim(poss_team_id, actual_quarter)

            release_moments, drop_reason, debug_info = lwc.find_moment_of_release(
                moments, shooter_id, poss_team_id, target_rim,
                rim_height=9.7,
                rim_xy_threshold=4.0,
                player_xyz_threshold=9.0,
                min_release_height=6.5,
                num_snapshots=3,
                moments_back=[3, 12],
                pbp_time_seconds=pbp_secs,
            )

            if release_moments is not None:
                break

        if release_moments is None:
            detail = None
            if drop_reason == "first_player_not_shooter" and debug_info:
                found_id = debug_info.get("found_player_id")
                if found_id:
                    detail = f"Nearest player: {get_name(found_id)}"
            entry["drop_reason"] = DROP_REASON_LABELS.get(drop_reason, drop_reason)
            entry["drop_detail"] = detail
            results.append(entry)
            continue

        # Successfully extracted
        entry["status"] = "extracted"
        results.append(entry)

    return results


# ── Per-game report ───────────────────────────────────────────────────────────
def process_game(json_path, pbp, label, lines):
    SEP = "─" * 90

    lines += ["", "=" * 90, f"GAME: {label}", "=" * 90]

    # 1. Run canonical extraction (also sets module-level rim globals via calibration)
    drop_stats = lwc.create_empty_drop_stats()
    cal_errors = lwc.create_empty_calibration_errors()
    shots = lwc.load_shot_attempts(json_path, pbp, drop_stats, cal_errors)

    # 2. Calibration check
    n_ok = len(cal_errors["calibrated_games"]) + len(cal_errors["calibrated_games_fallback"])
    if n_ok == 0:
        lines.append("ERROR: rim calibration failed — no shots extracted.")
        return
    method = "primary" if cal_errors["calibrated_games"] else "fallback"
    lines.append(f"Rim calibration: OK ({method} method)")
    lines.append("")

    # 3. Player name map from game JSON
    with open(json_path) as f:
        game = json.load(f)

    pname = {}
    if game["events"]:
        for side in ("home", "visitor"):
            for p in game["events"][0][side]["players"]:
                pid  = p["playerid"]
                name = f"{p.get('firstname','').strip()} {p.get('lastname','').strip()}".strip()
                pname[pid] = name

    def get_name(pid):
        try:
            return pname.get(int(pid), f"Player #{pid}")
        except Exception:
            return f"Player #{pid}"

    # 4. Extracted shots table
    n_made   = sum(1 for s in shots if s["made"])
    n_missed = len(shots) - n_made
    lines.append(f"EXTRACTED SHOTS  ({len(shots)} total  |  {n_made} made, {n_missed} missed)")
    lines.append(SEP)
    lines.append(f"  {'Shot':>5}  {'Time':>8}  {'Shooter':<30}  Result")
    lines.append(f"  {'─'*5}  {'─'*8}  {'─'*30}  {'─'*6}")

    for i, shot in enumerate(shots, 1):
        name = get_name(shot["shooter_id"])
        ts   = fmt_tracking(shot["game_clock"], shot["quarter"])
        res  = "MADE" if shot["made"] else "MISS"
        lines.append(f"  {i:>5}  {ts:>8}  {name:<30}  {res}")

    lines.append("")

    # 5. Per-shot outcome tracing for the unified drops table
    outcomes = trace_all_shot_outcomes(game, pbp, pname)
    drops = [o for o in outcomes if o["status"] == "dropped"]

    lines.append(f"DROPPED SHOTS  ({len(drops)} of {len(outcomes)} PBP shot events not extracted)")
    lines.append(SEP)

    if not drops:
        lines.append("  (none)")
    else:
        col_time   = 8
        col_name   = 30
        col_res    = 4
        col_reason = 36
        col_detail = 28
        col_desc   = 55

        hdr = (
            f"  {'Time':>{col_time}}  {'Shooter':<{col_name}}  {'Res':<{col_res}}  "
            f"{'Drop Reason':<{col_reason}}  {'Detail':<{col_detail}}  Description"
        )
        sep = (
            f"  {'─'*col_time}  {'─'*col_name}  {'─'*col_res}  "
            f"{'─'*col_reason}  {'─'*col_detail}  {'─'*col_desc}"
        )
        lines.append(hdr)
        lines.append(sep)

        for d in drops:
            res    = "MADE" if d["made"] else "MISS"
            reason = (d["drop_reason"] or "")[:col_reason]
            detail = (d["drop_detail"] or "")[:col_detail]
            desc   = d["description"][:col_desc]
            time_  = f"Q{d['quarter']} {d['time']:>5}"
            lines.append(
                f"  {time_:>{col_time}}  {d['shooter_name']:<{col_name}}  {res:<{col_res}}  "
                f"{reason:<{col_reason}}  {detail:<{col_detail}}  {desc}"
            )

    lines.append("")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("NBA Shot Extraction Analysis")
    print("=" * 60)

    pbp = get_pbp()

    lines = ["NBA SHOT EXTRACTION REPORT", "=" * 90, "Games analysed:"]
    for _, label in GAMES:
        lines.append(f"  • {label}")

    for filename, label in GAMES:
        print(f"\n{'─'*60}\n{label}")
        json_path = download_and_extract(filename)
        process_game(json_path, pbp, label, lines)

    report = "\n".join(lines)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"Report saved → {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
