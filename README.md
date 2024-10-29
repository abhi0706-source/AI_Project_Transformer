# AI Project - Maze Solver with Visualization

This project implements a maze solver using Breadth-First Search (BFS) and dynamically visualizes the exploration process. The program converts an image of a maze into a graph representation and uses BFS to find and display the shortest path from the start to the end.

## Folder Structure

AI_PROJECT/<br> 
├── generated_maze.png # Generated maze image used for testing <br>
<br>
├── grid_to_image.py # Converts grid representations to images <br>
<br>
├── main_code.py # Main code for maze solving and visualization <br>
<br>
├── map_parser.py # Parses maze maps from images or other sources <br>
<br>
├── map.png # Sample map image for testing
<br>
### File Descriptions

- *generated_maze.png*: A maze image used by main_code.py to perform maze-solving tests. This image should include start, end, walls, and open paths.

- *grid_to_image.py*: This script can convert grid representations of mazes into images. Useful for generating new maze images based on specific grid patterns.

- *main_code.py*: The primary code file, which contains functions to:
  - Parse an image of a maze and convert it into a matrix.
  - Construct a graph representation of the maze based on walkable paths.
  - Run the BFS algorithm to find the shortest path from the start to the end.
  - Visualize the exploration and final path dynamically in the maze.

- *map_parser.py*: Contains functions to parse maps (either from images or other formats) into a maze matrix that can be used by the solver.

- *map.png*: Another sample maze map that can be used for testing purposes.

## Main Code Details (main_code.py)

The main code performs the following steps:

1. *Image Parsing*:
   - Uses the image_to_maze_with_start_end function to convert a maze image into a grid where:
     - Black cells represent walls.
     - White cells represent walkable paths.
     - Red cells represent the start point.
     - Green cells represent the end point.

2. *Graph Construction*:
   - Uses the construct_graph function to create a graph where each node represents a cell, and edges connect walkable neighboring cells.

3. *BFS Visualization*:
   - The bfs_visualize function uses BFS to explore the maze. It visualizes each exploration step in real-time using matplotlib. 
   - Explored paths are displayed in blue, and the final path is highlighted in cyan.

4. *Maze Simulation*:
   - The simulate_maze_game function integrates the parsing, graph construction, and BFS functions to simulate the maze-solving process.
   - It takes the path to an image and visualizes the solution.

## How to Run

1. Place your maze image (e.g., generated_maze.png) in the same directory as main_code.py.
2. Adjust the cell_size parameter based on your maze image grid.
3. Run main_code.py to simulate the maze game with visualization.

### Example Command

```bash
python main_code.py

pip install pillow matplotlib numpy
