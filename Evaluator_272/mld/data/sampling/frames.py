from typing import Optional

import numpy as np
from numpy import ndarray as Array
import random


def get_frameix_from_data_index(num_frames: int,
                                request_frames: Optional[int],
                                sampling: str = "conseq",
                                sampling_step: int = 1) -> Array:
    nframes = num_frames

    if request_frames is None:
        frame_ix = np.arange(nframes)
    else:

        if request_frames > nframes:
            fair = False  # True
            if fair:
                # distills redundancy everywhere
                choices = np.random.choice(range(nframes),
                                           request_frames,
                                           replace=True)
                frame_ix = sorted(choices)
            else:
                # adding the last frame until done
                ntoadd = max(0, request_frames - nframes)
                lastframe = nframes - 1
                padding = lastframe * np.ones(ntoadd, dtype=int)
                frame_ix = np.concatenate((np.arange(0, nframes),
                                           padding))

        elif sampling in ["conseq", "random_conseq"]:
            step_max = (nframes - 1) // (request_frames - 1)
            if sampling == "conseq":
                if sampling_step == -1 or sampling_step * (request_frames - 1) >= nframes:
                    step = step_max
                else:
                    step = sampling_step
            elif sampling == "random_conseq":
                step = random.randint(1, step_max)

            lastone = step * (request_frames - 1)
            shift_max = nframes - lastone - 1
            shift = random.randint(0, max(0, shift_max - 1))
            frame_ix = shift + np.arange(0, lastone + 1, step)

        elif sampling == "random":
            choices = np.random.choice(range(nframes),
                                       request_frames,
                                       replace=False)
            frame_ix = sorted(choices)

        else:
            raise ValueError("Sampling not recognized.")

    return frame_ix
