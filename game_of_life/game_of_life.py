#!/usr/bin/env python3
"""
Conway's Game of Life — simple, fast, and interactive.

Key flags:
  --glider-gun           Seed with a Gosper glider gun near the left
  --hybrid-bc            Absorbing in X (left/right), periodic in Y (top/bottom)
  --click                Enable click-to-toggle cells (spacebar pauses/resumes)

Examples:
  # Indefinite glider source, vertical wrap, horizontal absorbing, interactive:
  python game_of_life.py --glider-gun --hybrid-bc --click --interval 50

  # Save a short GIF (no window):
  python game_of_life.py --glider-gun --hybrid-bc --steps 200 --save --out life.gif
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# ------------------------------- Patterns -------------------------------

def gosper_glider_gun(offset_x=0, offset_y=0):
    """Return (row, col) coords for the Gosper glider gun, offset from top-left."""
    gun = [
        (5,1),(5,2),(6,1),(6,2),
        (3,13),(3,14),(4,12),(4,16),(5,11),(5,17),(6,11),(6,15),(6,17),(6,18),
        (7,11),(7,17),(8,12),(8,16),(9,13),(9,14),
        (1,25),(2,23),(2,25),(3,21),(3,22),(4,21),(4,22),(5,21),(5,22),
        (6,23),(6,25),(7,25),
        (3,35),(3,36),(4,35),(4,36)
    ]
    return [(r + offset_y, c + offset_x) for (r, c) in gun]


# ------------------------------ Core Engine -----------------------------

class GameOfLife:
    def __init__(self, rows=60, cols=120, p=0.15, seed=None,
                 wrap_x=True, wrap_y=True,
                 glider_gun=False, offset_x=None, offset_y=None):
        rng = np.random.default_rng(seed)
        self.rows = rows
        self.cols = cols
        self.wrap_x = wrap_x  # periodic in columns (X)?
        self.wrap_y = wrap_y  # periodic in rows (Y)?

        # Init state
        if glider_gun:
            self.state = np.zeros((rows, cols), dtype=bool)
            if offset_x is None:
                offset_x = 1
            if offset_y is None:
                offset_y = max(1, rows // 2 - 5)
            for (r, c) in gosper_glider_gun(offset_x=offset_x, offset_y=offset_y):
                if 0 <= r < rows and 0 <= c < cols:
                    self.state[r, c] = True
        else:
            self.state = rng.random((rows, cols)) < p

    def neighbors_count(self, grid):
        """Count live neighbors under mixed boundary conditions."""
        g = grid.astype(np.uint8)

        # Case A: wrap both axes (torus)
        if self.wrap_x and self.wrap_y:
            n = (
                np.roll(np.roll(g,  1, axis=0),  1, axis=1) +
                np.roll(np.roll(g,  1, axis=0),  0, axis=1) +
                np.roll(np.roll(g,  1, axis=0), -1, axis=1) +
                np.roll(np.roll(g,  0, axis=0),  1, axis=1) +
                np.roll(np.roll(g,  0, axis=0), -1, axis=1) +
                np.roll(np.roll(g, -1, axis=0),  1, axis=1) +
                np.roll(np.roll(g, -1, axis=0),  0, axis=1) +
                np.roll(np.roll(g, -1, axis=0), -1, axis=1)
            ).astype(np.uint8)
            return n

        # Case B: absorb both axes
        if not self.wrap_x and not self.wrap_y:
            n = np.zeros_like(g, dtype=np.uint8)
            n[1:-1, 1:-1] = (
                g[ :-2,  :-2] + g[ :-2, 1:-1] + g[ :-2, 2: ] +
                g[1:-1,  :-2] +                  g[1:-1, 2: ] +
                g[2:  ,  :-2] + g[2:  , 1:-1] + g[2:  , 2: ]
            )
            return n

        # Case C: periodic Y, absorbing X  (your default hybrid)
        if self.wrap_y and not self.wrap_x:
            gp = np.pad(g, ((0,0),(1,1)), mode='constant', constant_values=0)  # pad columns
            up   = np.roll(gp,  1, axis=0)
            mid  = gp
            down = np.roll(gp, -1, axis=0)
            left, center, right = slice(0, -2), slice(1, -1), slice(2, None)
            n = (up[:, left] + up[:, center] + up[:, right] +
                 mid[:, left]               +  mid[:, right] +
                 down[:, left] + down[:, center] + down[:, right])
            return n.astype(np.uint8)

        # Case D: periodic X, absorbing Y
        if self.wrap_x and not self.wrap_y:
            gp = np.pad(g, ((1,1),(0,0)), mode='constant', constant_values=0)  # pad rows
            left  = np.roll(gp,  1, axis=1)
            mid   = gp
            right = np.roll(gp, -1, axis=1)
            top, center, bottom = slice(0, -2), slice(1, -1), slice(2, None)
            n = (left[top, :] + mid[top, :] + right[top, :] +
                              left[center, :] + right[center, :] +
                 left[bottom, :] + mid[bottom, :] + right[bottom, :])
            return n.astype(np.uint8)

        return np.zeros_like(g, dtype=np.uint8)

    def step(self):
        """Advance one timestep (B3/S23)."""
        grid = self.state
        n = self.neighbors_count(grid)
        survive = (grid & ((n == 2) | (n == 3)))
        born = (~grid & (n == 3))
        self.state = survive | born
        return self.state


# ------------------------------- Animation ------------------------------

def animate(rows=60, cols=120, p=0.15, seed=None, interval=60,
            steps=None, save=False, out="life.gif",
            wrap_x=True, wrap_y=True,
            glider_gun=False, offset_x=None, offset_y=None,
            click=False):
    game = GameOfLife(rows, cols, p, seed=seed,
                      wrap_x=wrap_x, wrap_y=wrap_y,
                      glider_gun=glider_gun, offset_x=offset_x, offset_y=offset_y)

    fig, ax = plt.subplots()
    im = ax.imshow(game.state, interpolation='nearest', origin='lower', cmap='gray', vmin=0, vmax=1)
    bc = ("wrap-x " if wrap_x else "absorb-x ") + ("wrap-y" if wrap_y else "absorb-y")
    title = "Conway's Game of Life"
    if glider_gun: title += " — Gosper Glider Gun"
    ax.set_title(f"{title}  [{bc}]")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_xticks([]); ax.set_yticks([])

    # Optional: click-to-toggle + space to pause
    paused = {"val": False}

    if click:
        def on_click(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            j = int(round(event.xdata))  # col
            i = int(round(event.ydata))  # row (origin='lower')
            if 0 <= i < game.rows and 0 <= j < game.cols:
                game.state[i, j] = ~game.state[i, j]
                im.set_data(game.state)
                fig.canvas.draw_idle()

        def on_key(event):
            if event.key == " ":
                paused["val"] = not paused["val"]

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        if not paused["val"]:
            game.step()
        im.set_data(game.state)
        return (im,)

    frames = steps if steps is not None else 200
    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

    if save:
        if not PIL_AVAILABLE:
            print("Pillow (PIL) not available; cannot save GIF. Install 'Pillow' to enable saving.", file=sys.stderr)
        else:
            # Render frames manually for reliable GIF creation
            images = []
            game_save = GameOfLife(rows, cols, p, seed=seed,
                                   wrap_x=wrap_x, wrap_y=wrap_y,
                                   glider_gun=glider_gun, offset_x=offset_x, offset_y=offset_y)
            for _ in range(frames):
                img = (game_save.state * 255).astype(np.uint8)
                images.append(Image.fromarray(img, mode='L').convert('P'))
                game_save.step()
            images[0].save(out, save_all=True, append_images=images[1:], optimize=False, duration=interval, loop=0)
            print(f"Saved GIF to: {out}")
        plt.close(fig)
    else:
        plt.show()


# --------------------------------- CLI ----------------------------------

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Conway's Game of Life — run & visualize (with optional GIF saving).")
    parser.add_argument("--rows", type=int, default=60, help="Grid rows (default: 60)")
    parser.add_argument("--cols", type=int, default=120, help="Grid cols (default: 120)")
    parser.add_argument("--p", type=float, default=0.15, help="Initial alive probability (when not using --glider-gun)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--interval", type=int, default=60, help="Animation frame interval in ms (default: 60)")
    parser.add_argument("--steps", type=int, default=None, help="Frames to animate (default: infinite window; 200 for saving)")
    parser.add_argument("--save", action="store_true", help="Save a GIF instead of showing a window (requires Pillow)")
    parser.add_argument("--out", type=str, default="life.gif", help="Output GIF filename (default: life.gif)")

    # Boundary controls
    parser.add_argument("--wrap-x", dest="wrap_x", action="store_true", help="Periodic in X (columns)")
    parser.add_argument("--no-wrap-x", dest="wrap_x", action="store_false", help="Absorbing in X (columns)")
    parser.add_argument("--wrap-y", dest="wrap_y", action="store_true", help="Periodic in Y (rows)")
    parser.add_argument("--no-wrap-y", dest="wrap_y", action="store_false", help="Absorbing in Y (rows)")
    parser.set_defaults(wrap_x=True, wrap_y=True)
    parser.add_argument("--hybrid-bc", action="store_true", help="Shortcut: absorbing X + periodic Y (sets --no-wrap-x --wrap-y)")

    # Pattern
    parser.add_argument("--glider-gun", action="store_true", help="Start with a Gosper glider gun (overrides random init)")
    parser.add_argument("--offset-x", type=int, default=None, help="Gun column offset (default: 1)")
    parser.add_argument("--offset-y", type=int, default=None, help="Gun row offset (default: vertically centered)")

    # Interactivity
    parser.add_argument("--click", action="store_true", help="Enable click-to-toggle editing; spacebar pauses/resumes")

    args = parser.parse_args(argv)

    if args.hybrid_bc:
        args.wrap_x = False
        args.wrap_y = True

    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    animate(rows=args.rows, cols=args.cols, p=args.p, seed=args.seed,
            interval=args.interval, steps=args.steps, save=args.save, out=args.out,
            wrap_x=args.wrap_x, wrap_y=args.wrap_y,
            glider_gun=args.glider_gun, offset_x=args.offset_x, offset_y=args.offset_y,
            click=args.click)


if __name__ == "__main__":
    main()
