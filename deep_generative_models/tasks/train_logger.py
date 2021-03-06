import os
import time

from csv import DictWriter
from logging import Logger

from typing import Optional, IO


class TrainLogger(object):
    PRINT_FORMAT = "epoch {:d}/{:d} {}: {:.05f} Time: {:.2f} s"
    CSV_COLUMNS = ["epoch", "metric_name", "metric_value", "time"]

    logger: Logger
    start_time: float
    output_file: Optional[IO]
    output_writer: Optional[DictWriter]

    def __init__(self, logger: Logger, output_path: str, append: bool = False) -> None:
        self.logger = logger

        if append and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            self.output_file = open(output_path, "a")
            self.output_writer = DictWriter(self.output_file, fieldnames=self.CSV_COLUMNS)
        else:
            self.output_file = open(output_path, "w")
            self.output_writer = DictWriter(self.output_file, fieldnames=self.CSV_COLUMNS)
            self.output_writer.writeheader()

        self.start_timer()

    def start_timer(self) -> None:
        self.start_time = time.time()

    def log(self, epoch: int, num_epochs: int, metric_name: str, metric_value: float) -> None:
        elapsed_time = time.time() - self.start_time

        self.output_writer.writerow({
            "epoch": epoch,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "time": elapsed_time
        })

        self.logger.info(self.PRINT_FORMAT.format(epoch,
                                                  num_epochs,
                                                  metric_name,
                                                  metric_value,
                                                  elapsed_time
                                                  ))

    def flush(self) -> None:
        self.output_file.flush()

    def close(self) -> None:
        self.output_file.close()

        # to release references just in case and to detect future invalid calls
        self.output_file = None
        self.output_writer = None
