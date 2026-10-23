from typing import Protocol

import numpy as np
import pandas as pd


class Session(Protocol):
    animal: str
    session: str
    t_start: float
    t_end: float
    eligible_intervals: np.ndarray
    units: pd.DataFrame

    @property
    def n_units(self) -> int: ...

    def make_view(self, start: float, duration: float, bin_size: float) -> tuple[np.ndarray, np.ndarray]: ...
