import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt


def gen_anim(data, cmap="PuOr"):
    fig, ax = plt.subplots()
    screen = ax.imshow(data[0], cmap=cmap)

    def update(i):
        img = screen.set_data(data[i])
        return screen,

    return FuncAnimation(
        fig, update, frames=len(data), blit=True, interval=1
    )

raster = np.load("rotating_wedge_events.npy").astype(int)
raster = raster[:, 1] - raster[:, 0]

anim = gen_anim(raster)
anim.save("rotating_events.gif", writer="imagemagick", fps=60)


frames = np.load("rotating_wedge_frames.npy").astype(int)

anim = gen_anim(frames)
anim.save("rotating_frames.gif", writer="imagemagick", fps=60)