from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def dummy_task(i):
    time.sleep(0.1)
    return i

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=6, max_tasks_per_child=32) as executor:
        futures = [executor.submit(dummy_task, i) for i in range(300)]
        for f in as_completed(futures):
            print(f.result())