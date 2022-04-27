"""
Generate animation of rotating wedges
Parameters used for sinabs vs slayer comparison:
size: 256
num_segments: 4
num_timesteps: 300
max_angle: 0.5 * pi
"""

import argparse
from typing import Optional
import numpy as np
import torch
try:
    from esim_torch import EventSimulator_torch
except (ImportError, ModuleNotFoundError):
    ESIM_AVAILABLE = False
else:
    ESIM_AVAILABLE = True


def draw_wedges(
    size: int = 256,
    num_segments: int = 4,
    num_timesteps: int = 300,
    max_angle: Optional[float] = None,
):
    """
    Wedges of alternating intensity (0 or 1) centered at origin.

    Parameter
    ---------
        size: int
            Size of the image(s) to be generated
        n-segs: int
            Number of segments, each consisting of two wedges of opposite intensity.
        num_timesteps: int
            Number of timesteps to rotate
        max_angle: float
            Angle up to which pattern will rotate. If None: Rotate until appears to be in
            initial position (up to rotational symmetry of figure)
    Returns
    -------
        np.array [size, size, num_timesteps]
            Intensity values along (y, x)-mesh, with additional
            0-dimension for different angles.
    """
    if max_angle is None:
        # Choose max angle as first angle at which rotated image is identical
        max_angle = 2 * np.pi / num_segments
    rotation = np.linspace(0, max_angle, num_timesteps, endpoint=False)
    rotation = rotation.reshape(-1, 1, 1)
    
    # x and y coordinates along a grid, with origin (0, 0) in the center
    grid = np.arange(size) - (size - 1) / 2
    x, y = np.meshgrid(grid, grid)

    # Polar angle of each point
    phi = np.arctan2(y, x)[np.newaxis, :, :]
    
    angles = phi + rotation
    
    # Split image in wedges based on angle
    wedge_numbers = np.floor_divide(num_segments * angles, np.pi)
    
    # Return either 0 or 1 based on wedge number
    return wedge_numbers % 2

def events_from_images(
    images, dt: float = 1e6, contrast_threshold=0.2, eps: float = 1e-6
):
    """ Convert images to events using esim_torch"""
    if not ESIM_AVAILABLE:
        raise RuntimeError(
            "This method requires esim_torch, which does not seem to be installed."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # - Make sure images are torch tensors
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float().to(device)
    # - Make sure images are on gpu
    images = images.to(device)

    timestamps = (torch.arange(len(images)) * dt).long().to(device)

    log_frames = torch.log(images + eps).float()
    esim = EventSimulator_torch(contrast_threshold_pos=contrast_threshold, contrast_threshold_neg=contrast_threshold)
    # Event generation
    return esim.forward(log_frames, timestamps)  
    

def raster_from_events(
    events,
    dt: float = 1e6,
    num_ts: Optional[float] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
):
    """Rasterize events"""
    # Unfortunately torch does not support N-dimensional histograms yet...
    events_tpyx = np.stack([events[s].cpu().numpy() for s in "tpyx"]).T
    if num_ts is None:
        # Infer number of time steps from timestamp of last event
        num_ts = int(events_tpyx[-1, -1] // dt)
    if height is None:
        # Infer height from events
        height = np.max(events_tpyx[:, 1]) + 1
    if width is None:
        # Infer width from events
        width = np.max(events_tpyx[:, 2]) + 1

    # Bins for time, polarity, y and x
    bins_t = np.arange(num_ts + 1) * dt
    bins_p = (-1.5, 0, 1.5)
    bins_y = np.arange(height + 1) - 0.5
    bins_x = np.arange(width + 1) - 0.5
    bins = (bins_t, bins_p, bins_y, bins_x)
    raster, *__ = np.histogramdd(events_tpyx, bins=bins)

    return raster

def generate_data(
    size: int = 256,
    num_segments: int = 4,
    num_timesteps: int = 300,
    max_angle: Optional[float] = None,
    save_path: Optional[str] = None,
    events: bool=True,
):

    print("Data generation")
    print("\tGenerating raw images...", end="")
    images = draw_wedges(size=size, num_segments=num_segments, num_timesteps=num_timesteps)
    print("\tDone.")

    if events:
        print("\tConverting images to events...", end="")
        events = events_from_images(images)
        print("\tDone.")
        print("\tConverting events to raster...", end="")
        data = raster_from_events(events, dt=1e6, height=size, width=size, num_ts=num_timesteps)
    else:
        data = images
    print("\tDone.")

    if save_path is not None:
        np.save(save_path, data.astype(np.uint8))
        print(f"Finished. Data has been stored as `{save_path}`.")
    else:
        print("Finished")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", "-s", type=int, default=256)
    parser.add_argument("--num_segments", "-n", type=int, default=4)
    parser.add_argument("--num_timesteps", "-t", type=int, default=300)
    parser.add_argument("--max_angle", "-a", type=float, default=None)
    parser.add_argument("--save_path", "-p", type=str, default=None)
    parser.add_argument(
        "--frames", "-f", dest="events", action="store_false")

    args = parser.parse_args()
    print(args.save_path)
    generate_data(
        size=args.size,
        num_segments=args.num_segments,
        num_timesteps=args.num_timesteps,
        max_angle=args.max_angle,
        save_path=args.save_path,
        events=args.events,
    )
