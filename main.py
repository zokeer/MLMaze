from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from QMaze import QMaze

maze = np.array([
    [1., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
    [0., 0., 1., 0., 0., 1., 0., 1., 1., 1.],
    [1., 1., 0., 1., 0., 1., 0., 0., 0., 1.],
    [1., 1., 0., 1., 0., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 0., 1., 1.]
])

visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
rat_mark = 0.5  # The current rat cell will be painted by gray 0.5
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

# Exploration factor
epsilon = 0.1


def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row, col in qmaze.visited:
        canvas[row, col] = 0.6
    rat_row, rat_col, _ = qmaze.state
    canvas[rat_row, rat_col] = 0.3  # rat cell
    canvas[nrows - 1, ncols - 1] = 0.9  # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img


qmaze = QMaze(maze)
canvas, reward, game_over = qmaze.act(DOWN)
print("reward=", canvas, reward, game_over)
canvas, reward, game_over = qmaze.act(LEFT)
print("reward=", reward)
canvas, reward, game_over = qmaze.act(DOWN)
print("reward=", reward)
show(qmaze)
