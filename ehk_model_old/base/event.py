from typing import List, Generic, TypeVar

from io import BufferedReader, BufferedWriter
import queue
import threading
import msgpack
import atexit

T = TypeVar('T')


class EventLogger(Generic[T]):

  def __init__(self, filename: str, batch_size=1000):
    self.filename = filename
    self.batch_size = batch_size

    self.queue = queue.Queue()
    self.stop_flag = threading.Event()
    self.worker_thread = threading.Thread(target=self._worker)
    self.worker_thread.daemon = True
    self.worker_thread.start()
    self.lock = threading.Lock()

    atexit.register(self.stop)

  def log_event(self, event: T):
    self.queue.put(event)

  def _worker(self):
    with open(self.filename, 'ab') as f:
      batch = []
      while not self.stop_flag.is_set() or not self.queue.empty():
        try:
          event = self.queue.get(timeout=1)
          batch.append(event)
          if len(batch) >= self.batch_size:
            self._write_batch(f, batch)
            batch = []
        except queue.Empty:
          if batch:
            self._write_batch(f, batch)
            batch = []

  def _write_batch(self, file: BufferedWriter, batch: List[T]):
    with self.lock:
      for event in batch:
        msgpack.dump(event, file)
      file.flush()

  def stop(self):
    with self.lock:
      if not self.stop_flag.is_set():
        self.stop_flag.set()
        self.worker_thread.join(timeout=5)

        # force write any remaining events
        if not self.queue.empty():
          with open(self.filename, 'ab') as f:
            while not self.queue.empty():
              batch = []
              while not self.queue.empty() and len(batch) < self.batch_size:
                batch.append(self.queue.get_nowait())
              self._write_batch(f, batch)


def read_msgpack_objects(file: BufferedReader):
  unpacker = msgpack.Unpacker(file, raw=False)
  for obj in unpacker:
    yield obj
