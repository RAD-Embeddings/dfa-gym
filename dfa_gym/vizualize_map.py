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


def visualize_map_minigrid(layout, figsize, cell_size=1, save_path=None):
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
                image_box = OffsetImage(image, zoom=0.05)
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
                        fontsize=24, color="black", weight="bold")

            elif content.islower():  # sync button
                if "a" in content:
                    color = "red"
                elif "b" in content:
                    color = "green"
                elif "c" in content:
                    color = "blue"
                elif "d" in content:
                    color = "pink"
                else:
                    raise ValueError
                # color = "crimson"
                if "#" in content:
                    ax.add_patch(patches.Rectangle(
                        (x, y), cell_size, cell_size,
                        facecolor=color, edgecolor="black", lw=1.5,
                        hatch="||", hatch_linewidth=3, fill=True  # diagonal stripes
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
                # for p in parts:
                #     if p.islower():
                #         ax.text(x + 0.5, y + 0.5, p,
                #                 ha="center", va="center",
                #                 fontsize=9, color="white")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    # layout = """
    # [ # ][ # ][ # ][ # ][ # ][   ][   ][   ][ 0 ][   ][   ][   ][ # ][ # ][ # ][ # ][ # ]
    # [ # ][ 0 ][   ][ 1 ][#,c][   ][ c ][   ][ A ][   ][ a ][   ][#,a][ 0 ][   ][ 2 ][ # ]
    # [ # ][   ][ 4 ][   ][#,c][   ][ c ][   ][   ][   ][ a ][   ][#,a][   ][ 8 ][   ][ # ]
    # [ # ][ 3 ][   ][ 2 ][#,c][   ][ c ][   ][ B ][   ][ a ][   ][#,a][ 6 ][   ][ 4 ][ # ]
    # [ # ][ # ][ # ][ # ][ # ][ 2 ][   ][   ][   ][   ][   ][ 3 ][ # ][ # ][ # ][ # ][ # ]
    # [ # ][ 5 ][   ][ 6 ][#,d][   ][ d ][   ][ C ][   ][ b ][   ][#,b][ 1 ][   ][ 3 ][ # ]
    # [ # ][   ][ 9 ][   ][#,d][   ][ d ][   ][   ][   ][ b ][   ][#,b][   ][ 9 ][   ][ # ]
    # [ # ][ 8 ][   ][ 7 ][#,d][   ][ d ][   ][ D ][   ][ b ][   ][#,b][ 7 ][   ][ 5 ][ # ]
    # [ # ][ # ][ # ][ # ][ # ][   ][   ][   ][ 1 ][   ][   ][   ][ # ][ # ][ # ][ # ][ # ]
    # """
    layout = """
    [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
    [ # ][ 0 ][   ][ 2 ][ # ][ 1 ][   ][ 3 ][ # ][ 0 ][   ][ 1 ][ # ][ 5 ][   ][ 6 ][ # ]
    [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    [ # ][ a ][ 8 ][ A ][ # ][ B ][ 9 ][ a ][#,a][ a ][ 4 ][ C ][#,a][ D ][ 9 ][ a ][ # ]
    [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    [ # ][ 6 ][   ][ 4 ][ # ][ 7 ][   ][ 5 ][ # ][ 3 ][   ][ 2 ][ # ][ 8 ][   ][ 7 ][ # ]
    [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
    """
    # layout = """
    # [ 0 ][   ][   ][   ][ # ][ # ][ # ][ # ][ # ]
    # [   ][   ][ a ][   ][#,a][ 0 ][   ][ 2 ][ # ]
    # [ A ][   ][ a ][   ][#,a][   ][ 8 ][   ][ # ]
    # [   ][   ][ a ][   ][#,a][ 6 ][   ][ 4 ][ # ]
    # [ 1 ][   ][   ][ 3 ][ # ][ # ][ # ][ # ][ # ]
    # [   ][   ][ b ][   ][#,b][ 1 ][   ][ 3 ][ # ]
    # [ B ][   ][ b ][   ][#,b][   ][ 9 ][   ][ # ]
    # [   ][   ][ b ][   ][#,b][ 7 ][   ][ 5 ][ # ]
    # [ 2 ][   ][   ][   ][ # ][ # ][ # ][ # ][ # ]
    # """
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

    visualize_map_minigrid(layout, figsize=(17,9), save_path="maps/4rooms_4agent.pdf")
