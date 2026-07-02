"""Output management for batched population synthesis results."""

import multiprocessing as mp
import queue
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from popkin.config.logger import get_logger
from popkin.utils import get_metallicity_str

logger = get_logger(__name__)


def _npy_to_csv(task):
    """Convert one npy batch into one CSV shard."""
    batch_file, csv_file, output_precision = task
    batch_file = Path(batch_file)
    csv_file = Path(csv_file)

    arr = np.load(batch_file, allow_pickle=True)
    df = pd.DataFrame.from_records(arr)
    df.to_csv(
        csv_file,
        header=True,
        index=False,
        float_format=f"%.{int(output_precision)}g",
    )

    return len(arr), int(getattr(arr, "nbytes", 0))


class OutputManager:
    """Batch output manager.

    Worker processes write intermediate npy batches. Final files are merged
    into csv, hdf5, or npy according to the requested output format.
    """

    def __init__(
            self,
            parallel,
            data_dir,
            output_format="csv",
            output_precision=6,
            batch_size=None,
            batch_max_bytes=64 * 1024 ** 2,
            writer_count=None,
            queue_maxsize=16,
            merge_workers=8,
            stop_signal_timeout=300,
    ):
        self.parallel = max(1, int(parallel))
        if writer_count is None:
            self.writer_count = self._default_writer_count()
        else:
            self.writer_count = max(1, int(writer_count))
        self.queue_maxsize = max(1, int(queue_maxsize))
        self.merge_workers = max(1, int(merge_workers))
        self.stop_signal_timeout = max(1, float(stop_signal_timeout))

        self.data_dir = Path(data_dir)
        self.output_format = output_format
        self.output_precision = int(output_precision)
        self.batch_size = None if batch_size is None else max(1, int(batch_size))
        self.batch_max_bytes = max(1, int(batch_max_bytes))

        self.queues = {}
        self.writer_processes = {}

    def _default_writer_count(self):
        if self.parallel < 4:
            return 1
        if self.parallel < 16:
            return 2
        if self.parallel < 32:
            return 3
        return 4
        # return min(32, max(1, self.parallel // 6 + 1))

    def start(self, targets, Z):
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[OutputManager] Configuration: format={self.output_format}, "
            f"targets={len(targets)}, writers/target={self.writer_count}, "
            f"queue_maxsize={self.queue_maxsize}, "
            f"batch_max={self.batch_max_bytes / 1024 ** 2:.0f}MB, "
            f"merge_workers={self.merge_workers}",
            extra={"console": True},
        )

        for target in targets:
            filename = f"{target['filename']}_{get_metallicity_str(Z)}"
            queues = []

            for writer_id in range(self.writer_count):
                queue = mp.Queue(maxsize=self.queue_maxsize)
                process = mp.Process(
                    target=self._writer_loop,
                    args=(filename, writer_id, queue),
                    name=f"OutputWriter-{filename}-{writer_id:02d}",
                )
                process.start()

                queues.append(queue)
                self.writer_processes[(filename, writer_id)] = process

            self.queues[filename] = queues

        return self.queues

    def _writer_loop(self, filename, writer_id, queue):
        batch = []
        batch_bytes = 0
        batch_num = 0

        try:
            while True:
                data = queue.get()

                if data is None:
                    if batch:
                        self._save_batch(filename, writer_id, batch, batch_num)
                    break

                data_bytes = int(getattr(data, "nbytes", 0))

                if batch and batch_bytes + data_bytes >= self.batch_max_bytes:
                    self._save_batch(filename, writer_id, batch, batch_num)
                    batch = []
                    batch_bytes = 0
                    batch_num += 1

                batch.append(data)
                batch_bytes += data_bytes

                hit_byte_limit = batch_bytes >= self.batch_max_bytes
                hit_count_limit = (
                    self.batch_size is not None
                    and len(batch) >= self.batch_size
                )

                if hit_byte_limit or hit_count_limit:
                    self._save_batch(filename, writer_id, batch, batch_num)
                    batch = []
                    batch_bytes = 0
                    batch_num += 1

        except Exception:
            logger.exception(
                f"[OutputManager] Writer failed: {filename}, writer={writer_id:02d}",
                extra={"console": True},
            )
            raise

    def _save_batch(self, filename, writer_id, batch, batch_num):
        t0 = time.perf_counter()
        combined = np.concatenate(batch)
        t_concat = time.perf_counter()

        output_file = self.data_dir / (
            f"{filename}_writer{writer_id:02d}_batch_{batch_num:06d}.npy"
        )
        np.save(output_file, combined, allow_pickle=True)
        t_save = time.perf_counter()

        logger.debug(
            f"[{filename}] writer={writer_id:02d}, batch={batch_num}: "
            f"{len(batch)} systems, {len(combined)} rows, "
            f"{combined.nbytes / 1024 ** 2:.1f} MB, "
            f"concat={t_concat - t0:.2f}s, save={t_save - t_concat:.2f}s",
            extra={"console": False},
        )

    def _check_writer_failures(self):
        for (filename, writer_id), process in self.writer_processes.items():
            if process.exitcode not in (None, 0):
                raise RuntimeError(
                    f"[OutputManager] Writer process exited with an error: "
                    f"{filename}, writer={writer_id:02d}, exitcode={process.exitcode}"
                )

    def _put_stop_signal(self, filename, writer_id, q):
        start_time = time.monotonic()
        last_warning_time = start_time

        while True:
            self._check_writer_failures()
            try:
                q.put(None, timeout=0.5)
                return
            except queue.Full:
                elapsed = time.monotonic() - start_time
                if elapsed >= self.stop_signal_timeout:
                    raise RuntimeError(
                        f"[OutputManager] Timed out sending stop signal: "
                        f"{filename}, writer={writer_id:02d}. "
                        f"The output queue stayed full for {self.stop_signal_timeout:.0f}s."
                    )

                now = time.monotonic()
                if now - last_warning_time >= 30:
                    logger.warning(
                        f"[OutputManager] Waiting to send stop signal: "
                        f"{filename}, writer={writer_id:02d}, elapsed={elapsed:.0f}s",
                        extra={"console": True},
                    )
                    last_warning_time = now

    def stop(self):
        for filename, queues in self.queues.items():
            for writer_id, q in enumerate(queues):
                logger.info(
                    f"[OutputManager] Sending stop signal: {filename}, writer={writer_id:02d}",
                    extra={"console": True},
                )
                self._put_stop_signal(filename, writer_id, q)

        for (filename, writer_id), process in self.writer_processes.items():
            process.join(timeout=60)

            if process.is_alive():
                raise RuntimeError(
                    f"[OutputManager] Writer process did not stop: "
                    f"{filename}, writer={writer_id:02d}, pid={process.pid}"
                )

            if process.exitcode != 0:
                raise RuntimeError(
                    f"[OutputManager] Writer process exited with an error: "
                    f"{filename}, writer={writer_id:02d}, exitcode={process.exitcode}"
                )

            logger.info(
                f"[OutputManager] Stopped: {filename}, writer={writer_id:02d}, exitcode={process.exitcode}",
                extra={"console": True},
            )

    def merge_all(self, keep_batches=False):
        for filename in self.queues.keys():
            batches = sorted(self.data_dir.glob(f"{filename}_writer*_batch_*.npy"))

            if not batches:
                continue

            if self.output_format == "csv":
                self._merge_csv(filename, batches, keep_batches)
            elif self.output_format == "hdf5":
                self._merge_hdf5(filename, batches, keep_batches)
            elif self.output_format == "npy":
                self._merge_npy(filename, batches, keep_batches)
            else:
                raise ValueError(
                    f"Unsupported output_format: '{self.output_format}'. Expected one of: 'csv', 'hdf5', 'npy'."
                )

    def _merge_csv(self, filename, batches, keep_batches):
        final_file = self.data_dir / f"{filename}.csv"
        csv_shards = [batch.with_suffix(".csv") for batch in batches]
        workers = min(self.parallel, self.merge_workers, len(batches))

        logger.info(
            f"[{filename}] Converting {len(batches)} batches -> CSV shards, "
            f"workers={workers}",
            extra={"console": True},
        )

        t0 = time.perf_counter()
        tasks = [
            (str(batch), str(csv_file), self.output_precision)
            for batch, csv_file in zip(batches, csv_shards)
        ]

        rows_total = 0
        bytes_total = 0
        with mp.Pool(processes=workers) as pool:
            for rows, nbytes in pool.imap_unordered(_npy_to_csv, tasks, chunksize=1):
                rows_total += rows
                bytes_total += nbytes

        t_convert = time.perf_counter()

        if final_file.exists():
            final_file.unlink()

        with final_file.open("wb") as dst:
            for i, csv_file in enumerate(csv_shards):
                with csv_file.open("rb") as src:
                    if i > 0:
                        src.readline()
                    shutil.copyfileobj(src, dst, length=1024 * 1024)

        t_concat = time.perf_counter()

        if not keep_batches:
            for file in batches + csv_shards:
                file.unlink()

        logger.info(
            f"[{filename}] Completed -> {final_file} | "
            f"rows={rows_total}, data={bytes_total / 1024 ** 2:.1f} MB, "
            f"convert={t_convert - t0:.2f}s, concat={t_concat - t_convert:.2f}s",
            extra={"console": True},
        )

    def _merge_hdf5(self, filename, batches, keep_batches):
        final_file = self.data_dir / f"{filename}.h5"
        first = True

        logger.info(f"[{filename}] Merging {len(batches)} batches into hdf5", extra={"console": True})

        for batch in batches:
            arr = np.load(batch, allow_pickle=True)
            df = pd.DataFrame.from_records(arr)
            df.to_hdf(
                final_file,
                key="data",
                mode="w" if first else "a",
                format="table",
                append=not first,
            )
            first = False

        if not keep_batches:
            for batch in batches:
                batch.unlink()

        logger.info(f"[{filename}] Completed -> {final_file}", extra={"console": True})

    def _merge_npy(self, filename, batches, keep_batches):
        final_file = self.data_dir / f"{filename}.npy"

        logger.info(f"[{filename}] Merging {len(batches)} batches into npy", extra={"console": True})

        arrays = [np.load(batch, allow_pickle=True) for batch in batches]
        np.save(final_file, np.concatenate(arrays), allow_pickle=True)

        if not keep_batches:
            for batch in batches:
                batch.unlink()

        logger.info(f"[{filename}] Completed -> {final_file}", extra={"console": True})

    def put(self, filename, data):
        self.queues[filename][0].put(data)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()
