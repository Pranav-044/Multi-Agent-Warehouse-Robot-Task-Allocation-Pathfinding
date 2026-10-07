

import heapq
import tkinter as tk
from tkinter import scrolledtext, ttk

from warehouse_sim import (
    Warehouse, Robot, WALL, PICKUP, DROPOFF,
    build_demo_env, manhattan, allocate_tasks, astar,
)

# ---------------------------------------------------------------------
# Color palette (matches the report / slides)
# ---------------------------------------------------------------------

COL_BG = "#FFFFFF"
COL_WALL = "#2B2B2B"
COL_FREE = "#F5F5F5"
COL_GRID_LINE = "#D1D5DB"
COL_FRONTIER = "#FBBC04"
COL_EXPANDED = "#CFE2FF"
COL_PATH = "#34A853"
COL_PICKUP = "#1A73E8"
COL_DROPOFF = "#D93025"
COL_PANEL = "#F3F4F6"
ROBOT_COLORS = ["#D93025", "#1A73E8", "#F9AB00", "#34A853", "#9334E6"]

CANVAS_SIZE = 560



EDGE1_LAYOUT = [
    "..........",
    ".####.####",
    "P....P....",
    ".####.####",
    "..........",
    "##########",
    "#D...P....",
    "##########",
    "..........",
    "..........",
]

EDGE2_LAYOUT = [
    "###########",
    "#.........#",
    "###########",
]



def astar_step_iter(env, start, goal, blocked=frozenset()):
    frontier = [(manhattan(start, goal), 0, start, [start])]
    best_g = {start: 0}
    expanded = []

    while frontier:
        f, g, node, path = heapq.heappop(frontier)
        yield {"type": "expand", "node": node, "g": g, "h": manhattan(node, goal), "f": f,
               "frontier": [it[2] for it in frontier], "expanded": list(expanded)}
        if node == goal:
            # goal is found at the moment it's popped, before being added to
            # the expanded set -- matches warehouse_sim.astar's node count
            yield {"type": "done", "path": path, "expanded": list(expanded),
                   "nodes_expanded": len(expanded)}
            return
        if node in expanded:
            continue
        expanded.append(node)
        for nxt in env.neighbors(node):
            if nxt in blocked and nxt != goal:
                continue
            new_g = g + 1
            if new_g < best_g.get(nxt, float("inf")):
                best_g[nxt] = new_g
                heapq.heappush(frontier, (new_g + manhattan(nxt, goal), new_g, nxt, path + [nxt]))
    yield {"type": "fail", "expanded": list(expanded)}


def simulate_tick_iter(env, robots, tasks, max_ticks=200, deadlock_patience=5):
    """Mirrors warehouse_sim.run_simulation's tick loop exactly, but
    yields a frame after every tick instead of printing at the end."""
    remaining_tasks = allocate_tasks(robots, tasks)
    delivered, total_tasks, stall_counter = 0, len(tasks), 0
    yield {"tick": 0, "event": None, "delivered": delivered, "total": total_tasks, "status": "running"}

    for tick in range(1, max_ticks + 1):
        other_positions = {r.pos for r in robots}
        for r in robots:
            if r.task and not r.path:
                blocked = other_positions - {r.pos}
                res = r.plan(env, blocked, astar)
                if not res.path:
                    yield {"tick": tick, "event": f"robot {r.id} FAILED to find a path (unreachable).",
                           "delivered": delivered, "total": total_tasks, "status": "running"}

        next_cell_of = {r.id: r.path[0] for r in robots if r.path}
        any_moved = False
        events = []
        for r in robots:
            reserved_by_others = {c for rid, c in next_cell_of.items() if rid != r.id}
            status = r.step(env, reserved_by_others)
            if status in ("MOVED", "DELIVERED"):
                any_moved = True
            if status == "DELIVERED":
                delivered += 1
                events.append(f"robot {r.id} DELIVERED task {r.task}")
                r.task, r.carrying, r.path = None, False, []
                remaining_tasks = allocate_tasks([r], remaining_tasks)

        yield {"tick": tick, "event": "; ".join(events) if events else None,
               "delivered": delivered, "total": total_tasks, "status": "running"}

        if total_tasks > 0 and delivered == total_tasks and all(r.task is None for r in robots):
            yield {"tick": tick, "event": f"All {total_tasks} tasks delivered. Simulation complete.",
                   "delivered": delivered, "total": total_tasks, "status": "done"}
            return

        active_robots = [r for r in robots if r.task is not None]
        if active_robots and not any_moved:
            stall_counter += 1
            if stall_counter >= deadlock_patience:
                yield {"tick": tick,
                       "event": f"DEADLOCK DETECTED among robots {[r.id for r in active_robots]} "
                                f"-- neither will yield. Simulation halted.",
                       "delivered": delivered, "total": total_tasks, "status": "deadlock"}
                return
        else:
            stall_counter = 0

    yield {"tick": max_ticks, "event": f"Max ticks reached. {delivered}/{total_tasks} delivered.",
           "delivered": delivered, "total": total_tasks, "status": "maxticks"}




class GridCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, width=CANVAS_SIZE, height=CANVAS_SIZE,
                          bg=COL_BG, highlightthickness=0, **kwargs)
        self.env = None
        self.cell = 40

    def cell_rect(self, x, y):
        x0 = self.ox + x * self.cell
        y0 = self.oy + y * self.cell
        return x0, y0, x0 + self.cell, y0 + self.cell

    def draw_static(self, env: Warehouse):
        self.delete("all")
        self.env = env
        self.cell = min((CANVAS_SIZE - 20) / env.width, (CANVAS_SIZE - 20) / env.height)
        self.ox = (CANVAS_SIZE - env.width * self.cell) / 2
        self.oy = (CANVAS_SIZE - env.height * self.cell) / 2
        for y in range(env.height):
            for x in range(env.width):
                x0, y0, x1, y1 = self.cell_rect(x, y)
                base = env.grid[y][x]
                if base == WALL:
                    self.create_rectangle(x0, y0, x1, y1, fill=COL_WALL, outline=COL_WALL, tags="static")
                else:
                    self.create_rectangle(x0, y0, x1, y1, fill=COL_FREE, outline=COL_GRID_LINE, tags="static")
                    if base == PICKUP:
                        self.create_text((x0 + x1) / 2, (y0 + y1) / 2, text="P",
                                          fill=COL_PICKUP, font=("Helvetica", int(self.cell * 0.4), "bold"),
                                          tags=("static", "static_text"))
                    elif base == DROPOFF:
                        self.create_text((x0 + x1) / 2, (y0 + y1) / 2, text="D",
                                          fill=COL_DROPOFF, font=("Helvetica", int(self.cell * 0.4), "bold"),
                                          tags=("static", "static_text"))

    def clear_search(self):
        self.delete("search")

    def clear_robots(self):
        self.delete("robot")

    def highlight_cells(self, cells, color, tag_extra=""):
        for (x, y) in cells:
            x0, y0, x1, y1 = self.cell_rect(x, y)
            pad = self.cell * 0.06
            self.create_rectangle(x0 + pad, y0 + pad, x1 - pad, y1 - pad,
                                   fill=color, outline="", tags=("search", tag_extra))
        if self.find_withtag("static_text"):
            self.tag_raise("static_text")
        if self.find_withtag("robot"):
            self.tag_raise("robot")

    def draw_path_line(self, path):
        pts = []
        for (x, y) in path:
            x0, y0, x1, y1 = self.cell_rect(x, y)
            pts.extend([(x0 + x1) / 2, (y0 + y1) / 2])
        if len(pts) >= 4:
            self.create_line(*pts, fill=COL_PATH, width=max(3, self.cell * 0.12),
                              capstyle=tk.ROUND, joinstyle=tk.ROUND, tags="search")
        for (x, y) in path:
            x0, y0, x1, y1 = self.cell_rect(x, y)
            r = self.cell * 0.12
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=COL_PATH, outline="", tags="search")

    def draw_robots(self, robots):
        self.clear_robots()
        for i, r in enumerate(robots):
            x0, y0, x1, y1 = self.cell_rect(*r.pos)
            pad = self.cell * 0.12
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            self.create_oval(x0 + pad, y0 + pad, x1 - pad, y1 - pad,
                              fill=color, outline="#222222", width=1.5, tags="robot")
            self.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=r.id, fill="white",
                              font=("Helvetica", int(self.cell * 0.4), "bold"), tags="robot")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Agent Warehouse — 23CSE401 Case Study")
        self.configure(bg=COL_BG)
        self.resizable(False, False)

        self.speed_ms = tk.IntVar(value=140)
        self._after_id = None

        self._build_layout()
        self.scene_search_demo() 

  
    def _build_layout(self):
        outer = tk.Frame(self, bg=COL_BG)
        outer.pack(padx=14, pady=14)

        left = tk.Frame(outer, bg=COL_BG)
        left.grid(row=0, column=0, sticky="n")
        self.canvas = GridCanvas(left)
        self.canvas.pack()

        legend = tk.Frame(left, bg=COL_BG)
        legend.pack(pady=(8, 0), anchor="w")
        self._legend_item(legend, COL_FRONTIER, "frontier")
        self._legend_item(legend, COL_EXPANDED, "expanded")
        self._legend_item(legend, COL_PATH, "path / robot")
        self._legend_item(legend, COL_WALL, "wall")

        right = tk.Frame(outer, bg=COL_PANEL, padx=14, pady=14)
        right.grid(row=0, column=1, sticky="n", padx=(14, 0))

        tk.Label(right, text="Scenarios", bg=COL_PANEL, font=("Helvetica", 13, "bold")).pack(anchor="w")

        btn_opts = dict(width=30, anchor="w", relief="flat", bg="#FFFFFF",
                         activebackground="#E8F0FE", padx=10, pady=6, cursor="hand2")
        buttons = [
            ("▶  Animate A* Search", self.scene_search_demo),
            ("▶  Working Case 1 — 2 robots", self.scene_working_case_1),
            ("▶  Working Case 2 — 3 robots (congested)", self.scene_working_case_2),
            ("▶  Edge Case 1 — unreachable goal", self.scene_edge_case_1),
            ("▶  Edge Case 2 — corridor deadlock", self.scene_edge_case_2),
        ]
        for label, cmd in buttons:
            tk.Button(right, text=label, command=cmd, **btn_opts).pack(fill="x", pady=3)

        tk.Label(right, text="Speed", bg=COL_PANEL, font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(14, 0))
        tk.Scale(right, from_=250, to=10, orient="horizontal", variable=self.speed_ms,
                  showvalue=False, bg=COL_PANEL, troughcolor="#D1D5DB", length=260).pack(fill="x")
        row = tk.Frame(right, bg=COL_PANEL)
        row.pack(fill="x")
        tk.Label(row, text="slower", bg=COL_PANEL, fg="#6B7280", font=("Helvetica", 9)).pack(side="left")
        tk.Label(row, text="faster", bg=COL_PANEL, fg="#6B7280", font=("Helvetica", 9)).pack(side="right")

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(right, textvariable=self.status_var, bg=COL_PANEL, fg=COL_PICKUP,
                  font=("Helvetica", 11, "bold"), wraplength=280, justify="left").pack(anchor="w", pady=(14, 4))

        tk.Label(right, text="Event Log", bg=COL_PANEL, font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.log = scrolledtext.ScrolledText(right, width=38, height=14, font=("Consolas", 9),
                                              bg="#0F172A", fg="#7CFC9A", insertbackground="white",
                                              relief="flat")
        self.log.pack(pady=(4, 0))
        self.log.configure(state="disabled")

    def _legend_item(self, parent, color, text):
        f = tk.Frame(parent, bg=COL_BG)
        f.pack(side="left", padx=(0, 12))
        tk.Canvas(f, width=14, height=14, bg=COL_BG, highlightthickness=0).pack(side="left")
        sw = tk.Canvas(f, width=14, height=14, bg=COL_BG, highlightthickness=0)
        sw.create_rectangle(1, 1, 13, 13, fill=color, outline="#999999")
        sw.pack(side="left")
        tk.Label(f, text=" " + text, bg=COL_BG, font=("Helvetica", 9)).pack(side="left")


    def log_clear(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.configure(state="disabled")

    def log_append(self, text):
        self.log.configure(state="normal")
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _cancel_pending(self):
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None


    def scene_search_demo(self):
        self._cancel_pending()
        env = build_demo_env()
        self.canvas.draw_static(env)
        self.canvas.clear_robots()
        self.log_clear()
        self.status_var.set("Animating A* search: (0,0) → pickup (1,2) ...")
        self.log_append("=== A* SEARCH ANIMATION ===")
        self.log_append("start=(0,0)  goal=(1,2)")
        self._gen = astar_step_iter(env, (0, 0), (1, 2))
        self._advance_search()

    def _advance_search(self):
        try:
            frame = next(self._gen)
        except StopIteration:
            return
        self.canvas.clear_search()
        if frame["type"] == "expand":
            self.canvas.highlight_cells(frame["expanded"], COL_EXPANDED)
            self.canvas.highlight_cells(frame["frontier"], COL_FRONTIER)
            self.canvas.highlight_cells([frame["node"]], "#FDE68A")
            self.log_append(f"expand {frame['node']}  g={frame['g']} h={frame['h']} f={frame['f']}")
            self._after_id = self.after(self.speed_ms.get(), self._advance_search)
        elif frame["type"] == "done":
            self.canvas.highlight_cells(frame["expanded"], COL_EXPANDED)
            self.canvas.draw_path_line(frame["path"])
            n = frame["nodes_expanded"]
            self.status_var.set(f"GOAL reached — {n} nodes expanded, path length {len(frame['path']) - 1}.")
            self.log_append(f">>> GOAL REACHED. {n} nodes expanded. Path length {len(frame['path']) - 1} steps.")
        elif frame["type"] == "fail":
            self.canvas.highlight_cells(frame["expanded"], COL_EXPANDED)
            self.status_var.set("NO PATH FOUND — goal unreachable.")
            self.log_append(">>> NO PATH FOUND. Frontier exhausted.")


    def _start_sim(self, env, robots, tasks, title):
        self._cancel_pending()
        self.canvas.draw_static(env)
        self.canvas.clear_search()
        self.canvas.draw_robots(robots)
        self.log_clear()
        self.log_append(f"=== {title} ===")
        self.status_var.set(title)
        self._robots = robots
        self._gen = simulate_tick_iter(env, robots, tasks)
        self._advance_sim()

    def _advance_sim(self):
        try:
            frame = next(self._gen)
        except StopIteration:
            return
        self.canvas.draw_robots(self._robots)
        if frame["event"]:
            self.log_append(f"tick {frame['tick']}: {frame['event']}")
        self.status_var.set(f"Tick {frame['tick']}  —  Delivered {frame['delivered']}/{frame['total']}")
        if frame["status"] == "running":
            self._after_id = self.after(self.speed_ms.get(), self._advance_sim)
        else:
            tag = {"done": "COMPLETE", "deadlock": "DEADLOCK", "maxticks": "STOPPED"}[frame["status"]]
            self.status_var.set(f"{tag} at tick {frame['tick']} — {frame['delivered']}/{frame['total']} delivered.")

    def scene_working_case_1(self):
        env = build_demo_env()
        robots = [Robot("A", (0, 0)), Robot("B", (9, 9))]
        tasks = [((1, 2), (1, 6)), ((5, 2), (5, 6))]
        self._start_sim(env, robots, tasks, "WORKING CASE 1 — Two robots, two independent tasks")

    def scene_working_case_2(self):
        env = build_demo_env()
        robots = [Robot("A", (0, 0)), Robot("B", (9, 0)), Robot("C", (0, 9))]
        tasks = [((1, 2), (1, 6)), ((5, 2), (5, 6)), ((6, 2), (6, 6))]
        self._start_sim(env, robots, tasks, "WORKING CASE 2 — Three robots, congested")

    def scene_edge_case_1(self):
        env = Warehouse(EDGE1_LAYOUT)
        robots = [Robot("A", (0, 0))]
        tasks = [((1, 2), (1, 6))]
        self._start_sim(env, robots, tasks, "EDGE CASE 1 — Unreachable dropoff (sealed by walls)")

    def scene_edge_case_2(self):
        env = Warehouse(EDGE2_LAYOUT)
        robots = [Robot("A", (1, 1)), Robot("B", (9, 1))]
        robots[0].assign(((9, 1), (9, 1)))
        robots[1].assign(((1, 1), (1, 1)))
        self._start_sim(env, robots, [], "EDGE CASE 2 — Head-on deadlock in a 1-wide corridor")


if __name__ == "__main__":
    app = App()
    app.mainloop()
