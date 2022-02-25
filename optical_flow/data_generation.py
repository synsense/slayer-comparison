"""
Generate animation of rotating wedges
Parameters used for sinabs vs slayer comparison:
size: 256
n_segs: 4
num_timesteps: 300
max_angle: 0.5 * pi
"""

def draw_wedges(
    size: int = 256,
    n_segs: int = 4,
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
            dimension for different angles.
    """
    if max_angle is None:
        # Choose max angle as first angle at which rotated image is identical
        max_angle = 2 * np.pi / n_segs
    rotation = np.linspace(0, max_angle, num_timesteps, endpoint=False)
    rotation = rotation.reshape(1, 1, -1)
    
    # x and y coordinates along a grid, with origin (0, 0) in the center
    grid = np.arange(size) - (size - 1) / 2
    x, y = np.meshgrid(grid, grid)

    # Polar angle of each point
    phi = np.arctan2(y, x)[:, :, np.newaxis]
    
    angles = phi + rotation
    
    # Split image in wedges based on angle
    wedge_numbers = np.floor_divide(n_segs * angles, np.pi)
    
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
    # - Make sure images are torch tensors
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float().to(DEVICE)
    # - Make sure images are on gpu
    images = images.to(DEVICE)

    timestamps = (torch.arange(len(images)) * dt).long().to(DEVICE)

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

if __name__ == "__main__":
    size: 256
    n_segs: 4
    num_timesteps: 300
    max_angle: 0.5 * pi
    
    images = draw_wedges(size=size, n_segs=n_segs, num_timesteps=num_timesteps)
    events = events_from_images(images)
    raster = raster_from_events(events, dt=1e6, height=size, width=size, num_ts=num_timesteps)
    
    np.save("raster.npy", raster)