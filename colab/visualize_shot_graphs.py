"""
Visualize shot graphs on a basketball court.
Shows player positions, ball, graph edges, and temporal trails.
Usage: python colab/visualize_shot_graphs.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from pathlib import Path

# ─── Court drawing ────────────────────────────────────────────────────────────

COURT_COLOR  = '#333333'
PAINT_FILL   = '#e8e0d0'
COURT_FILL   = '#f7f3ec'

# 3pt arc geometry: basket at (bx, 25), radius R=23.75, corners at y=3 and y=47
_R = 23.75
_CORNER_Y_LO, _CORNER_Y_HI = 3.0, 47.0
_CORNER_DX = np.sqrt(_R**2 - (25 - _CORNER_Y_LO)**2)  # ≈ 8.95 ft
# Arc angles for right basket (center at (bx,25)):
#   bottom junction at angle atan2(y_lo-25, -_CORNER_DX) ≈ 247.8°
#   top    junction at angle atan2(y_hi-25, -_CORNER_DX) ≈ 112.2°
_ARC_THETA_LO = np.degrees(np.arctan2(_CORNER_Y_LO - 25, -_CORNER_DX))  # ≈ -112.2
_ARC_THETA_HI = np.degrees(np.arctan2(_CORNER_Y_HI - 25, -_CORNER_DX))  # ≈  112.2


def draw_court(ax):
    lw, color = 1.5, COURT_COLOR

    # Court surface
    ax.add_patch(patches.Rectangle((0, 0), 94, 50, linewidth=lw,
                                    edgecolor=color, facecolor=COURT_FILL, zorder=0))
    # Center line
    ax.plot([47, 47], [0, 50], color=color, lw=lw, zorder=1)

    # ── Right basket (baseline at x=94, basket at x=88.75) ─────────────────
    bx_r = 88.75
    # Corner 3 lines: horizontal, from baseline to arc junction
    ax.plot([94, bx_r - _CORNER_DX], [_CORNER_Y_LO, _CORNER_Y_LO], color=color, lw=lw, zorder=1)
    ax.plot([94, bx_r - _CORNER_DX], [_CORNER_Y_HI, _CORNER_Y_HI], color=color, lw=lw, zorder=1)
    # 3pt arc: left-facing portion (theta1 to theta2, counterclockwise)
    ax.add_patch(patches.Arc((bx_r, 25), 2*_R, 2*_R, angle=0,
                              theta1=_ARC_THETA_HI, theta2=_ARC_THETA_LO + 360,
                              color=color, lw=lw, zorder=1))
    # Paint (19 ft deep from baseline, 16 ft wide)
    ax.add_patch(patches.Rectangle((75, 17), 19, 16, linewidth=lw,
                                    edgecolor=color, facecolor=PAINT_FILL, zorder=1))
    # FT circle (center at free-throw line, 15 ft from basket = x=73.75, but key is 19ft so x=75)
    ax.add_patch(patches.Circle((75, 25), 6, fill=False, color=color, lw=lw, zorder=1))
    # Restricted area (4 ft radius, facing away from baseline = left-facing)
    ax.add_patch(patches.Arc((bx_r, 25), 8, 8, angle=0,
                              theta1=90, theta2=270, color=color, lw=lw, zorder=1))
    # Backboard & rim
    ax.plot([94, 94], [22, 28], color=color, lw=2.5, zorder=1)
    ax.add_patch(plt.Circle((bx_r, 25), 0.75, color='#c0392b', fill=False, lw=2, zorder=2))

    # ── Left basket (baseline at x=0, basket at x=5.25) ────────────────────
    bx_l = 5.25
    ax.plot([0, bx_l + _CORNER_DX], [_CORNER_Y_LO, _CORNER_Y_LO], color=color, lw=lw, zorder=1)
    ax.plot([0, bx_l + _CORNER_DX], [_CORNER_Y_HI, _CORNER_Y_HI], color=color, lw=lw, zorder=1)
    # 3pt arc: right-facing portion
    arc_lo_r = np.degrees(np.arctan2(_CORNER_Y_LO - 25,  _CORNER_DX))  # ≈ -67.8
    arc_hi_r = np.degrees(np.arctan2(_CORNER_Y_HI - 25,  _CORNER_DX))  # ≈  67.8
    ax.add_patch(patches.Arc((bx_l, 25), 2*_R, 2*_R, angle=0,
                              theta1=arc_lo_r, theta2=arc_hi_r,
                              color=color, lw=lw, zorder=1))
    ax.add_patch(patches.Rectangle((0, 17), 19, 16, linewidth=lw,
                                    edgecolor=color, facecolor=PAINT_FILL, zorder=1))
    ax.add_patch(patches.Circle((19, 25), 6, fill=False, color=color, lw=lw, zorder=1))
    # Restricted area (4 ft radius, facing away from baseline = right-facing)
    ax.add_patch(patches.Arc((bx_l, 25), 8, 8, angle=0,
                              theta1=270, theta2=90, color=color, lw=lw, zorder=1))
    ax.plot([0, 0], [22, 28], color=color, lw=2.5, zorder=1)
    ax.add_patch(plt.Circle((bx_l, 25), 0.75, color='#c0392b', fill=False, lw=2, zorder=2))


# ─── Shot visualization ───────────────────────────────────────────────────────

OFFENSE_COLOR = '#d62728'   # matplotlib red
DEFENSE_COLOR = '#1f77b4'   # matplotlib blue
BALL_COLOR    = '#ff7f0e'   # matplotlib orange
SHOOTER_RING  = '#2ca02c'   # matplotlib green
EDGE_COLOR    = '#7f7f7f'
TRAIL_ALPHA   = 0.45


def plot_shot(ax, shot, game_label=''):
    draw_court(ax)

    off_pos    = shot['offense_position']
    def_pos    = shot['defense_position']
    ball_pos   = shot['ball_position']
    rim        = shot['target_rim']
    shooter_id = shot['shooter_id']
    names      = shot.get('player_names', {})
    prev       = shot.get('previous_moments', [])

    # ── Temporal trail ──────────────────────────────────────────────────────
    for moment in prev:
        for row in moment['offense_position']:
            ax.plot(row[2], row[3], 'o', color=OFFENSE_COLOR,
                    alpha=TRAIL_ALPHA, ms=7, zorder=3)
        for row in moment['defense_position']:
            ax.plot(row[2], row[3], 's', color=DEFENSE_COLOR,
                    alpha=TRAIL_ALPHA, ms=7, zorder=3)
        bx, by = moment['ball_position'][0], moment['ball_position'][1]
        ax.plot(bx, by, 'o', color=BALL_COLOR, alpha=TRAIL_ALPHA, ms=6, zorder=3)

    # ── Within-timestep graph edges ─────────────────────────────────────────
    all_nodes = ([(r[2], r[3]) for r in off_pos] +
                 [(r[2], r[3]) for r in def_pos] +
                 [(ball_pos[0], ball_pos[1])])
    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            ax.plot([all_nodes[i][0], all_nodes[j][0]],
                    [all_nodes[i][1], all_nodes[j][1]],
                    color=EDGE_COLOR, alpha=0.15, lw=0.8, zorder=2)

    # ── Defense ─────────────────────────────────────────────────────────────
    for row in def_pos:
        pid, x, y = row[1], row[2], row[3]
        ax.plot(x, y, 's', color=DEFENSE_COLOR, ms=12, zorder=5,
                markeredgecolor='white', markeredgewidth=1.0)
        label = names.get(str(pid), str(pid)).split()[-1]
        ax.text(x, y + 1.5, label, ha='center', va='bottom',
                fontsize=7, color=DEFENSE_COLOR, fontweight='bold', zorder=6)

    # ── Offense ─────────────────────────────────────────────────────────────
    for row in off_pos:
        pid, x, y = row[1], row[2], row[3]
        is_shooter = (pid == shooter_id)
        ax.plot(x, y, 'o', color=OFFENSE_COLOR,
                ms=14 if is_shooter else 12, zorder=5,
                markeredgecolor=SHOOTER_RING if is_shooter else 'white',
                markeredgewidth=2.5 if is_shooter else 1.0)
        label = names.get(str(pid), str(pid)).split()[-1]
        ax.text(x, y + 1.5, label, ha='center', va='bottom',
                fontsize=7, color=OFFENSE_COLOR, fontweight='bold', zorder=6)

    # ── Ball ────────────────────────────────────────────────────────────────
    bx, by = ball_pos[0], ball_pos[1]
    ax.plot(bx, by, 'o', color=BALL_COLOR, ms=11, zorder=7,
            markeredgecolor='#7f3f00', markeredgewidth=1.2)

    # ── Zoom to the relevant half-court ─────────────────────────────────────
    mid_x = np.mean([r[2] for r in off_pos])
    if mid_x > 47:
        ax.set_xlim(43, 97)
    else:
        ax.set_xlim(-3, 51)
    ax.set_ylim(-3, 53)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Title ───────────────────────────────────────────────────────────────
    shooter_name = names.get(str(shooter_id), 'Unknown')
    dist = ((bx - rim[0])**2 + (by - rim[1])**2)**0.5
    sc = shot['shot_clock']
    sc_str = f'SC {int(sc):.0f}s' if sc is not None else ''
    result = 'MADE' if shot['made'] else 'MISSED'
    result_color = '#2ca02c' if shot['made'] else '#d62728'

    gc = int(shot['game_clock'])
    gc_str = f'{gc // 60}:{gc % 60:02d}'
    ax.set_title(
        f'{shooter_name}  —  {game_label}\n'
        f'Q{shot["quarter"]}  |  {gc_str}  |  {sc_str}  |  {dist:.1f} ft',
        fontsize=12, fontweight='bold', pad=10
    )
    ax.text(0.5, -0.02, result, transform=ax.transAxes,
            ha='center', va='top', fontsize=13, fontweight='bold', color=result_color)

    # ── Legend ──────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=OFFENSE_COLOR,
               markersize=9, label='Offense'),
        Line2D([0], [0], marker='o', color=SHOOTER_RING, markerfacecolor=OFFENSE_COLOR,
               markersize=11, markeredgewidth=2.5, label='Shooter'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=DEFENSE_COLOR,
               markersize=9, label='Defense'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=BALL_COLOR,
               markersize=9, label='Ball'),
        Line2D([0], [0], color=EDGE_COLOR, alpha=0.6, lw=1.5, label='Graph edges'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=OFFENSE_COLOR,
               alpha=0.4, markersize=7, label='Temporal trail'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8,
              framealpha=0.9, edgecolor='#cccccc')


# ─── Main ─────────────────────────────────────────────────────────────────────

def pick_shots(shots_file, n_made=2, n_missed=1):
    data = json.load(open(shots_file))
    made   = [s for s in data if s['made']]
    missed = [s for s in data if not s['made']]

    def dist(s):
        b, r = s['ball_position'], s['target_rim']
        return ((b[0]-r[0])**2 + (b[1]-r[1])**2)**0.5

    made.sort(key=dist)
    missed.sort(key=dist)
    made_picks   = [made[len(made)//4], made[len(made)//2]][:n_made]
    missed_picks = [missed[len(missed)//2]][:n_missed]
    return made_picks + missed_picks


def main():
    base    = Path(__file__).parent / 'data' / 'shots_data'
    out_dir = Path(__file__).parent.parent / 'report' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    games = {
        '0021500436': 'MIA vs NOP',
        '0021500468': 'GSW vs SAC',
    }

    for game_id, game_label in games.items():
        shots = pick_shots(base / f'shots_{game_id}.json', n_made=2, n_missed=1)

        for i, shot in enumerate(shots):
            fig, ax = plt.subplots(figsize=(8, 7))
            plot_shot(ax, shot, game_label=game_label)
            plt.tight_layout()

            result_tag = 'made' if shot['made'] else 'missed'
            shooter_name = shot['player_names'].get(str(shot['shooter_id']), 'shot')
            slug = shooter_name.split()[-1].lower()
            out_path = out_dir / f'shot_{game_id}_{i+1}_{slug}_{result_tag}.png'
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'Saved → {out_path}')

    print('Done.')


if __name__ == '__main__':
    main()
