import os
import os.path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ipywidgets import interactive, interactive_output, interact_manual
import ipywidgets as widgets
import cv2
import time
import mediapy
from tqdm.notebook import tqdm
import pyarrow
mpl.rcParams['figure.figsize'] = [16.0, 12.0]

import tonic
import tonic.transforms as transforms

brisbane_event_traverses = [
    'dvs_vpr_2020-04-21-17-03-03_no_hot_pixels_nobursts_denoised.feather',
    'dvs_vpr_2020-04-22-17-24-21_no_hot_pixels_nobursts_denoised.feather',
    # 'dvs_vpr_2020-04-24-15-12-03_no_hot_pixels_nobursts_denoised.feather',
    # 'dvs_vpr_2020-04-28-09-14-11_no_hot_pixels_nobursts_denoised.feather',
    # 'dvs_vpr_2020-04-29-06-20-23_no_hot_pixels_nobursts_denoised.feather',
]

qcr_traverses = [
    'bags_2021-08-19-08-25-42_denoised.feather', # side-facing, slow
    'bags_2021-08-19-08-28-43_denoised.feather', # side-facing, slow
    'bags_2021-08-19-09-45-28_denoised.feather', # side-facing, slow
    # 'bags_2021-08-20-09-52-59_denoised.feather', # down-facing, slow
    # 'bags_2021-08-20-09-49-58_denoised.feather', # down-facing, slow
    # 'bags_2021-08-20-10-19-45_denoised.feather', # side-facing, fast
]

path_to_qcr_event_files = './Data/'

traverse = qcr_traverses[0]

event_stream = pd.read_feather(os.path.join(path_to_qcr_event_files, traverse))

im_width, im_height = int(event_stream['x'].max() + 1), int(event_stream['y'].max() + 1)

ordering = "txyp"

x_index = ordering.find("x")
y_index = ordering.find("y")
t_index = ordering.find("t")
p_index = ordering.find("p")

sensor_size = (im_width, im_height)
sensor_size

event_stream

event_stream_numpy = np.copy(event_stream.to_numpy(np.uint64))

print(f'Time duration: {(event_stream_numpy[-1, 0] - event_stream_numpy[0, 0]) / 10e5 :.2f}s')

display_timestep = 1.0 / 30.0
history_time = 2.0 / 30.0

display_timestep_ms = int(display_timestep * 10e5)
history_time_ms = int(history_time * 10e5)

target_times = np.arange(event_stream_numpy[0, t_index] + history_time_ms, event_stream_numpy[-1, t_index], display_timestep_ms, dtype=np.uint64)

# plt.figure(1)

def get_start_end_indices(target_time_idx):
    target_time = target_times[target_time_idx]

    start_time = np.uint64(target_time - history_time_ms)
    end_time = np.uint64(target_time)

    start_idx = event_stream_numpy[:, t_index].searchsorted(start_time)
    end_idx = event_stream_numpy[:, t_index].searchsorted(end_time)

    return start_idx, end_idx

def plot_event_frame(target_time_idx):
    measure_time_start = time.perf_counter()
    start_idx, end_idx = get_start_end_indices(target_time_idx)
    target_time = target_times[target_time_idx]

    out_image = np.ones((im_height, im_width, 3), dtype=np.uint8) * 255
    events = event_stream_numpy[start_idx:end_idx]
    for t, x, y, p in events:
        out_image[y, x] = (0, 0, 255) if p == 0 else (255, 0, 0)

    text_to_print = f"idx: {target_time_idx}; t: {target_time / 10e5: .2f}; len: {len(events)}"
    cv2.putText(out_image, text_to_print, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

    scale = 2
    mediapy.show_image(out_image, width=im_width*scale, height=im_height*scale)

play = widgets.Play(
    value=0,
    min=0,
    max=len(target_times),
    step=1,
    interval=display_timestep * 1000,
    description="Press play",
    disabled=False
)
slider = widgets.IntSlider(max=len(target_times))
widgets.jslink((play, 'value'), (slider, 'value'))
# widgets.HBox([play, slider])

interactive_plot = interactive(plot_event_frame, target_time_idx=play)
# output = interactive_plot.children[-1]
# output.layout.height = '350px'
# interactive_plot
display(widgets.VBox([slider, interactive_plot]))

transform_example = transforms.Compose([
    transforms.DropEvent(0.9),
    transforms.ToFrame(sensor_size=sensor_size, ordering=ordering, time_window=30000),
])

place_number = 2
time_start = event_stream_numpy[0, 0] + place_number * 10e5
time_end = event_stream_numpy[0, 0] + (place_number + 1) * 10e5

start_idx = np.searchsorted(event_stream_numpy[:, 0], time_start)
end_idx = np.searchsorted(event_stream_numpy[:, 0], time_end)

events = np.copy(event_stream_numpy[start_idx:end_idx])

out = transform_example(events)