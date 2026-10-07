"""Generates the figures used in the LaTeX report."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from warehouse_sim import (build_demo_env, astar, bfs, WALL, PICKUP, DROPOFF,
                            DEMO_LAYOUT, complexity_benchmark)

OUT = "../report/figures"


def draw_grid(env, path=None, start=None, goal=None, title="", fname="grid.png"):
    fig, ax = plt.subplots(figsize=(5, 5))
    for y in range(env.height):
        for x in range(env.width):
            c = env.grid[y][x]
            color = "#2b2b2b" if c == WALL else "#f5f5f5"
            ax.add_patch(Rectangle((x, env.height - 1 - y), 1, 1,
                                    facecolor=color, edgecolor="#cccccc", linewidth=0.5))
            if c == PICKUP:
                ax.text(x + 0.5, env.height - 1 - y + 0.5, "P", ha="center", va="center",
                        fontsize=11, color="#1a73e8", fontweight="bold")
            elif c == DROPOFF:
                ax.text(x + 0.5, env.height - 1 - y + 0.5, "D", ha="center", va="center",
                        fontsize=11, color="#d93025", fontweight="bold")
    if path:
        xs = [p[0] + 0.5 for p in path]
        ys = [env.height - 1 - p[1] + 0.5 for p in path]
        ax.plot(xs, ys, color="#34a853", linewidth=3, marker="o", markersize=5, zorder=5)
    if start:
        ax.add_patch(Rectangle((start[0], env.height - 1 - start[1]), 1, 1,
                                facecolor="none", edgecolor="#34a853", linewidth=3))
    if goal:
        ax.add_patch(Rectangle((goal[0], env.height - 1 - goal[1]), 1, 1,
                                facecolor="none", edgecolor="#d93025", linewidth=3))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{fname}", dpi=160)
    plt.close()


def draw_benchmark(bench, fname="complexity.png"):
    ns = [b["n"] for b in bench]
    astar_nodes = [b["astar_nodes_avg"] for b in bench]
    bfs_nodes = [b["bfs_nodes_avg"] for b in bench]
    astar_time = [b["astar_time_avg"] * 1000 for b in bench]
    bfs_time = [b["bfs_time_avg"] * 1000 for b in bench]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ns, astar_nodes, marker="o", label="A*", color="#1a73e8")
    axes[0].plot(ns, bfs_nodes, marker="s", label="BFS", color="#d93025")
    axes[0].set_xlabel("Grid size N (N x N)")
    axes[0].set_ylabel("Avg. nodes expanded")
    axes[0].set_title("Nodes Expanded vs Grid Size")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ns, astar_time, marker="o", label="A*", color="#1a73e8")
    axes[1].plot(ns, bfs_time, marker="s", label="BFS", color="#d93025")
    axes[1].set_xlabel("Grid size N (N x N)")
    axes[1].set_ylabel("Avg. runtime (ms)")
    axes[1].set_title("Runtime vs Grid Size")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/{fname}", dpi=160)
    plt.close()


if __name__ == "__main__":
    env = build_demo_env()

    res_a = astar(env, (0, 0), (1, 2))
    draw_grid(env, path=res_a.path, start=(0, 0), goal=(1, 2),
              title="A* Path: (0,0) -> (1,2)", fname="astar_path.png")

    res_b = bfs(env, (0, 0), (1, 2))
    draw_grid(env, path=res_b.path, start=(0, 0), goal=(1, 2),
              title="BFS Path: (0,0) -> (1,2)", fname="bfs_path.png")

    draw_grid(env, title="Warehouse Environment Layout", fname="environment.png")

    with open("benchmark_results.json") as f:
        bench = json.load(f)
    draw_benchmark(bench)

    print("Figures written to", OUT)
