# Multi-Agent Warehouse Robot Task Allocation and Pathfinding

This repository contains a discrete 2D grid-world simulation of a multi-agent warehouse system. It simulates cooperative robots navigating through a warehouse, picking up items at designated pickup locations (`P`), and delivering them to dropoff locations (`D`), all while avoiding walls (`#`) and colliding with each other.

The project demonstrates search algorithms (A* vs. BFS) and multi-agent coordination.

## 🚀 Features

*   **A* Search Algorithm:** Implements an informed A* Search using the Manhattan distance heuristic, optimizing pathfinding on a 4-way movement grid.
*   **Multi-Agent Coordination:** Dynamically allocates tasks to the nearest idle robots based on Manhattan distance.
*   **Collision Avoidance:** Robots use a reservation system to avoid stepping into the same cell simultaneously.
*   **Deadlock Detection:** Automatically detects when robots are in a gridlock and halts the simulation.
*   **Interactive GUI:** A Tkinter-based user interface to visualize the search algorithms, track robot paths, and simulate different layout scenarios.
*   **Benchmarking:** Built-in benchmarking to compare the performance (runtime and nodes expanded) of A* vs. Breadth-First Search (BFS).

## 📂 Project Structure

*   `warehouse_sim.py`: The core simulation engine. Contains the Grid environment, Robot logic, A*/BFS pathfinding algorithms, and the main simulation loop.
*   `gui_demo.py`: A Tkinter GUI to visualize the warehouse, animate A* pathfinding, and step through different working/edge case scenarios.
*   `make_figures.py`: A utility script using `matplotlib` to generate figures and benchmark graphs.

## 🛠️ Setup and Usage

### Prerequisites
*   Python 3.x
*   `matplotlib` (Only if you want to generate benchmark graphs)

### Running the Visualizer
To launch the interactive Tkinter GUI:
```bash
python gui_demo.py
```
From the GUI, you can select different scenarios such as "Working Case", "Edge Case - Unreachable Goal", and "Edge Case - Corridor Deadlock".

### Running the Simulator (Headless) & Benchmarks
To run the terminal-based text output and generate the complexity benchmarks:
```bash
python warehouse_sim.py
```

### Generating Figures
To generate the visualizations for paths and performance graphs:
```bash
python make_figures.py
```

## 🧠 Why Manhattan Distance?
The A* search uses Manhattan distance (`|x1 - x2| + |y1 - y2|`) because the robots are restricted to 4-way movement (up, down, left, right). Manhattan distance is **admissible** (it never overestimates the cost) and gives the tightest lower bound for this grid, making A* explore the fewest nodes possible. Euclidean distance would underestimate the true grid distance, making the algorithm slower.
