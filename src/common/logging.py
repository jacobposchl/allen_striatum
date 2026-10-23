"""Flushed per-run logs and stage timing without a training dependency."""
from datetime import datetime, timezone
import json
import logging
import sys
from time import perf_counter
from uuid import uuid4


class StageTimer:
    def __init__(self, synchronize=None):
        self.synchronize = synchronize
        self.last = perf_counter()

    def mark(self):
        if self.synchronize is not None:
            self.synchronize()
        now = perf_counter()
        elapsed, self.last = now - self.last, now
        return elapsed


class RunLog:
    def __init__(self, directory):
        self.attempt = uuid4().hex[:12]
        self.logger = logging.Logger(f"nuclr.{self.attempt}", level=logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (logging.StreamHandler(sys.stdout),
                        logging.FileHandler(directory / "training.log", encoding="utf-8")):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.steps = (directory / "steps.jsonl").open("a", encoding="utf-8")

    def info(self, message):
        self.logger.info(message)

    def exception(self, message):
        self.logger.exception(message)

    def step(self, values):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "attempt": self.attempt, **values}
        self.steps.write(json.dumps(record, allow_nan=False) + "\n")
        self.steps.flush()

    def close(self):
        self.steps.close()
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
