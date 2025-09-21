import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import re


def parse_map(map_lines):
    """Parses the ASCII map into a 2D grid of cells."""
    grid = []
    for line in map_lines:
        cells = re.findall(r"\[(.*?)\]", line)
        if cells:
            grid.append([c.strip() for c in cells])
    return grid


def visualize_map_minigrid(layout, cell_size=1, figsize=(6, 6), save_path=None):
    map_lines = layout.splitlines()
    grid = parse_map(map_lines)
    n_rows, n_cols = len(grid), len(grid[0])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.axis("off")

    for r in range(n_rows):
        for c in range(n_cols):
            content = grid[r][c]
            x, y = c, n_rows - r - 1

            # background
            ax.add_patch(patches.Rectangle(
                (x, y), cell_size, cell_size,
                facecolor="lightgray", edgecolor="white", lw=1
            ))
    for r in range(n_rows):
        for c in range(n_cols):
            content = grid[r][c]
            x, y = c, n_rows - r - 1

            if not content:
                continue

            if content == "#":  # wall
                ax.add_patch(patches.Rectangle(
                    (x, y), cell_size, cell_size,
                    facecolor="dimgray", edgecolor="black", lw=1.5
                ))

            elif content.isupper():  # agents
                image = plt.imread('robot.png')
                image_box = OffsetImage(image, zoom=0.023)
                ab = AnnotationBbox(image_box, (x + 0.5, y + 0.5), frameon=False)
                ax.add_artist(ab)

                # ax.text(x + 0.5, y + 0.5, "8",
                #         ha="center", va="center",
                #         fontsize=14, weight="bold")

            elif content.isdigit():  # tokens
                ax.add_patch(patches.Circle(
                    (x + 0.5, y + 0.5), 0.4,
                    facecolor="gold", edgecolor="orange", lw=1.5
                ))
                ax.text(x + 0.5, y + 0.5, content,
                        ha="center", va="center",
                        fontsize=11, color="black", weight="bold")

            elif content.islower():  # sync button
                if "a" in content:
                    color = "red"
                elif "b" in content:
                    color = "green"
                elif "c" in content:
                    color = "blue"
                else:
                    raise ValueError
                # color = "crimson"
                if "#" in content:
                    ax.add_patch(patches.Rectangle(
                        (x, y), cell_size, cell_size,
                        facecolor=color, edgecolor="black", lw=1.5,
                        hatch="|||", hatch_linewidth=2, fill=True  # diagonal stripes
                    ))
                else:
                    ax.add_patch(patches.Rectangle(
                        (x, y), cell_size, cell_size,
                        facecolor=color, edgecolor="black", lw=1.5
                    ))

            elif "," in content:  # door like "#,a"
                parts = [p.strip() for p in content.split(",")]
                ax.add_patch(patches.Rectangle(
                    (x, y), cell_size, cell_size,
                    facecolor="firebrick", edgecolor="black", lw=1.5
                ))
                for p in parts:
                    if p.islower():
                        ax.text(x + 0.5, y + 0.5, p,
                                ha="center", va="center",
                                fontsize=9, color="white")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    layout = """
    [ 0 ][   ][   ][   ][ # ][ # ][ # ][ # ][ # ]
    [   ][   ][ a ][   ][#,a][ 0 ][   ][ 2 ][ # ]
    [ A ][   ][ a ][   ][#,a][   ][ 8 ][   ][ # ]
    [   ][   ][ a ][   ][#,a][ 6 ][   ][ 4 ][ # ]
    [ 1 ][   ][   ][ 3 ][ # ][ # ][ # ][ # ][ # ]
    [   ][   ][ b ][   ][#,b][ 1 ][   ][ 3 ][ # ]
    [ B ][   ][ b ][   ][#,b][   ][ 9 ][   ][ # ]
    [   ][   ][ b ][   ][#,b][ 7 ][   ][ 5 ][ # ]
    [ 2 ][   ][   ][   ][ # ][ # ][ # ][ # ][ # ]
    """
    # layout = """
    # [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
    # [ # ][ 0 ][   ][ 2 ][ # ][ 1 ][   ][ 3 ][ # ]
    # [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    # [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    # [ # ][ a ][ 8 ][ A ][#,a][ B ][ 9 ][ a ][ # ]
    # [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    # [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    # [ # ][ 6 ][   ][ 4 ][ # ][ 7 ][   ][ 5 ][ # ]
    # [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
    # """

    visualize_map_minigrid(layout, save_path="maps/layout1.pdf")
