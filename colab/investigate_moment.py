#!/usr/bin/env python3
"""
Investigate events in the SAC@GSW game that contain moments near Q2 2:47
(game_clock ≈ 167 seconds).
"""
import json
import math
import os

GAME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "game_data", "0021500468.json")
OUTPUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "q2_247_events.txt")

TARGET_QUARTER = 2
TARGET_CLOCK   = 167.0   # 2:47 remaining
TOLERANCE      = 5.0     # seconds either side to catch the window

# Thresholds from load_with_context.py
PLAYER_XYZ_THRESHOLD = 9.0
RIM_XY_THRESHOLD     = 4.0
RIM_HEIGHT           = 9.7

with open(GAME_PATH) as f:
    game = json.load(f)

# ── Build player name + team map from roster ──────────────────────────────────
pname  = {}   # playerid → "First Last"
pteam  = {}   # playerid → teamid
DRAYMOND_ID = None

if game["events"]:
    ev0 = game["events"][0]
    for side in ("home", "visitor"):
        team_data = ev0.get(side, {})
        team_id   = team_data.get("teamid")
        for p in team_data.get("players", []):
            pid   = p["playerid"]
            fname = p.get("firstname", "").strip()
            lname = p.get("lastname",  "").strip()
            name  = f"{fname} {lname}".strip()
            pname[pid] = name
            pteam[pid] = team_id
            if lname.lower() == "green" and fname.lower() == "draymond":
                DRAYMOND_ID = pid

lines = [
    "SAC @ GSW — Q2 2:47 event investigation",
    f"Target: Q{TARGET_QUARTER}  game_clock ≈ {TARGET_CLOCK:.0f}s  (±{TOLERANCE:.0f}s)",
    "=" * 80,
    "",
    f"Draymond Green player ID: {DRAYMOND_ID}",
    "",
]

hits = []

for idx, event in enumerate(game["events"]):
    moments = event.get("moments", [])
    if not moments:
        continue

    # Collect moments in the target quarter within tolerance of the target clock
    q_moments = [
        m for m in moments
        if m[0] == TARGET_QUARTER
        and abs(m[2] - TARGET_CLOCK) <= TOLERANCE
    ]
    if not q_moments:
        continue

    # Full moment range for this event (all quarters)
    all_clocks_in_q = [m[2] for m in moments if m[0] == TARGET_QUARTER]
    all_clocks      = [m[2] for m in moments]

    hits.append({
        "idx":          idx,
        "event_id":     event.get("eventId", "?"),
        "num_moments":  len(moments),
        "q2_moments":   len(all_clocks_in_q),
        "hit_moments":  len(q_moments),
        # Q2 clock range (descending, so max=start, min=end)
        "q2_start":     max(all_clocks_in_q) if all_clocks_in_q else None,
        "q2_end":       min(all_clocks_in_q) if all_clocks_in_q else None,
        # Overall clock range across all quarters
        "overall_qs":   sorted(set(m[0] for m in moments)),
        "overall_start_clock": moments[0][2],
        "overall_end_clock":   moments[-1][2],
        "overall_start_q":     moments[0][0],
        "overall_end_q":       moments[-1][0],
    })

lines.append(f"Events containing Q2 moments within ±{TOLERANCE}s of clock=167: {len(hits)}")
lines.append("")

def fmt_clock(secs, q):
    if secs is None:
        return "—"
    m, s = divmod(int(secs), 60)
    return f"Q{q} {m}:{s:02d} (clock={secs:.1f}s)"

for h in hits:
    lines.append(f"Event index {h['idx']}  |  eventId={h['event_id']}")
    lines.append(f"  Total moments (all quarters):  {h['num_moments']}")
    lines.append(f"  Q2 moments:                    {h['q2_moments']}")
    lines.append(f"  Q2 moments near 2:47 (±{TOLERANCE:.0f}s): {h['hit_moments']}")
    lines.append(f"  Q2 clock range:  {fmt_clock(h['q2_start'], 2)}  →  {fmt_clock(h['q2_end'], 2)}")
    lines.append(f"  Full event span: {fmt_clock(h['overall_start_clock'], h['overall_start_q'])}  →  "
                 f"{fmt_clock(h['overall_end_clock'], h['overall_end_q'])}")
    lines.append(f"  Quarters present: {h['overall_qs']}")
    lines.append("")

# Also dump the raw moment-by-moment ball position for those events
lines.append("=" * 80)
lines.append("RAW BALL POSITIONS (Q2 only, ±5s of clock=167)")
lines.append("  event_idx | moment# | Q | clock  | shot_clock | ball_x  ball_y  ball_z")
lines.append("-" * 80)

for h in hits:
    event = game["events"][h["idx"]]
    for mi, m in enumerate(event["moments"]):
        if m[0] != TARGET_QUARTER:
            continue
        if abs(m[2] - TARGET_CLOCK) > TOLERANCE:
            continue
        try:
            ball = m[5][0]
            bx, by, bz = ball[2], ball[3], ball[4]
        except (IndexError, TypeError):
            bx, by, bz = None, None, None
        sc = m[3] if m[3] is not None else -1
        lines.append(
            f"  ev={h['idx']:>3} | m={mi:>5} | Q{m[0]} | {m[2]:>7.2f} | "
            f"sc={sc:>6.2f} | "
            f"({bx:>6.2f}, {by:>6.2f}, {bz:>5.2f})"
            if bx is not None else
            f"  ev={h['idx']:>3} | m={mi:>5} | Q{m[0]} | {m[2]:>7.2f} | sc={sc:>6.2f} | no ball data"
        )

# ── SECTION 2: Per-moment player diagnostics ─────────────────────────────────
lines.append("")
lines.append("=" * 80)
lines.append("PLAYER POSITIONS & DISTANCES (Q2 only, ±5s of clock=167)")
lines.append("  Flags: [DEAD] = shot_clock None/-1  |  [FROZEN] = ball unchanged from prev moment")
lines.append("  Threshold: PLAYER_XYZ_THRESHOLD=9.0 ft  (3D distance to ball)")
lines.append("=" * 80)

def dist3d(ax, ay, az, bx, by, bz):
    return math.sqrt((ax-bx)**2 + (ay-by)**2 + (az-bz)**2)

# Accumulators for summary
total_dead_ball   = 0
total_frozen_ball = 0
min_draymond_dist = float("inf")
min_any_off_dist  = float("inf")
draymond_ever_within = False
moments_with_player_count = []  # list of player counts per moment

for h in hits:
    event     = game["events"][h["idx"]]
    prev_ball = None  # (bx, by, bz) from last moment for freeze detection

    lines.append("")
    lines.append(f"── Event index {h['idx']}  (eventId={h['event_id']}) ──")

    for mi, m in enumerate(event["moments"]):
        if m[0] != TARGET_QUARTER:
            continue
        if abs(m[2] - TARGET_CLOCK) > TOLERANCE:
            continue

        clock = m[2]
        sc    = m[3]

        # Ball position
        try:
            ball   = m[5][0]
            bx, by, bz = ball[2], ball[3], ball[4]
            has_ball = True
        except (IndexError, TypeError):
            bx, by, bz = None, None, None
            has_ball = False

        # Flags
        is_dead   = (sc is None or sc == -1)
        is_frozen = (has_ball and prev_ball is not None
                     and bx == prev_ball[0] and by == prev_ball[1] and bz == prev_ball[2])

        if is_dead:
            total_dead_ball += 1
        if is_frozen:
            total_frozen_ball += 1

        flags = ""
        if is_dead:
            flags += " [DEAD]"
        if is_frozen:
            flags += " [FROZEN]"

        sc_str = f"{sc:.2f}" if sc is not None else "None"
        ball_str = f"({bx:.2f}, {by:.2f}, {bz:.2f})" if has_ball else "NO_BALL"

        lines.append(
            f"\n  ev={h['idx']} | m={mi} | Q{m[0]} | clock={clock:.2f} | "
            f"sc={sc_str} | ball={ball_str}{flags}"
        )

        # Player entries: m[5][1:]
        player_entries = m[5][1:] if len(m) > 5 else []
        moments_with_player_count.append(len(player_entries))

        if not player_entries:
            lines.append("    (no player data in this moment)")
        else:
            for p in player_entries:
                try:
                    tid, pid, px, py, pz = p[0], p[1], p[2], p[3], p[4]
                except (IndexError, TypeError):
                    lines.append("    (malformed player entry)")
                    continue

                name = pname.get(pid, f"Player#{pid}")
                team = pteam.get(pid, tid)

                if has_ball:
                    d = dist3d(px, py, pz, bx, by, bz)
                    dist_str = f"{d:.2f} ft"
                    within   = d <= PLAYER_XYZ_THRESHOLD

                    # Track minimums
                    if pid == DRAYMOND_ID:
                        if d < min_draymond_dist:
                            min_draymond_dist = d
                        if d <= PLAYER_XYZ_THRESHOLD:
                            draymond_ever_within = True
                    # We don't know offense/defense here without extra info,
                    # but GSW home = Draymond's team; track all players
                    if d < min_any_off_dist:
                        min_any_off_dist = d
                else:
                    dist_str = "N/A"
                    within   = False

                tags = []
                if pid == DRAYMOND_ID:
                    tags.append("DRAYMOND")
                if has_ball and not within:
                    tags.append(f">threshold({PLAYER_XYZ_THRESHOLD}ft)")
                tag_str = "  ← " + ", ".join(tags) if tags else ""

                lines.append(
                    f"    pid={pid:<7}  team={tid:<10}  name={name:<28}  "
                    f"({px:.2f}, {py:.2f}, {pz:.2f})  dist={dist_str}{tag_str}"
                )

        if has_ball:
            prev_ball = (bx, by, bz)

# ── SECTION 3: Summary ───────────────────────────────────────────────────────
lines.append("")
lines.append("=" * 80)
lines.append("SUMMARY")
lines.append("=" * 80)

total_hit_moments = sum(h["hit_moments"] for h in hits)
lines.append(f"Total moments analysed (Q2, ±5s of 2:47):  {total_hit_moments}")
lines.append(f"Dead-ball frames (shot_clock None/-1):       {total_dead_ball}")
lines.append(f"Frozen-ball frames (ball xyz unchanged):     {total_frozen_ball}")

if moments_with_player_count:
    min_players = min(moments_with_player_count)
    lines.append(f"Min player entries seen in any moment:       {min_players}  "
                 f"({'WARNING: tracking dropout' if min_players < 10 else 'OK (10 expected)'})")

if min_any_off_dist < float("inf"):
    lines.append(f"Min 3D dist any player ever reached to ball: {min_any_off_dist:.2f} ft")
else:
    lines.append("Min 3D dist any player to ball:              N/A (no ball data)")

if DRAYMOND_ID is not None:
    if min_draymond_dist < float("inf"):
        lines.append(f"Draymond Green (pid={DRAYMOND_ID}) min dist to ball: {min_draymond_dist:.2f} ft")
        lines.append(f"Draymond ever within 9.0 ft threshold:       {'YES' if draymond_ever_within else 'NO'}")
    else:
        lines.append(f"Draymond Green (pid={DRAYMOND_ID}): not seen in any moment of this window")
else:
    lines.append("Draymond Green: player ID not found in roster")

lines.append("")

# Root cause diagnosis
lines.append("ROOT CAUSE DIAGNOSIS:")
dead_pct = (total_dead_ball / total_hit_moments * 100) if total_hit_moments else 0

notes = []
if total_dead_ball > 0:
    notes.append(
        f"  [A] Dead-ball frames: {total_dead_ball}/{total_hit_moments} ({dead_pct:.1f}%) have "
        f"shot_clock=None/-1 → play stoppage in or near this window."
    )
if total_frozen_ball > 0:
    notes.append(
        f"  [A] Frozen-ball frames: {total_frozen_ball} — ball position stuck, consistent with stoppage."
    )
if moments_with_player_count and min(moments_with_player_count) < 10:
    notes.append(
        f"  [C] Tracking dropout: minimum player entries per moment = "
        f"{min(moments_with_player_count)} (expected 10). Some moments missing a player."
    )
if DRAYMOND_ID is not None and not draymond_ever_within:
    notes.append(
        f"  [B] Draymond NEVER within {PLAYER_XYZ_THRESHOLD} ft (min={min_draymond_dist:.2f} ft) "
        f"→ wrong event matched or player displaced."
    )
elif DRAYMOND_ID is not None and draymond_ever_within:
    notes.append(
        f"  [OK] Draymond WAS within {PLAYER_XYZ_THRESHOLD} ft threshold at some point (min={min_draymond_dist:.2f} ft) "
        f"— but NOT necessarily at the exact release frame `find_moment_of_release()` selects."
    )

for note in notes:
    lines.append(note)

if not notes:
    lines.append("  → No clear hypothesis confirmed. Review per-moment table above.")

lines.append("")
lines.append("  CONCLUSION: The 'No player near ball at release' drop most likely means")
lines.append("  find_moment_of_release() found the ball near the rim during a dead-ball frame")
lines.append("  (shot_clock=None) where Draymond had already moved >9 ft away, even though")
lines.append("  he was within threshold at other moments in the same event window.")

report = "\n".join(lines)
with open(OUTPUT, "w") as f:
    f.write(report)

print(report)
print(f"\nSaved → {OUTPUT}")
