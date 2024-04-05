from functools import wraps
from tqdm import tqdm
import asyncio
import time

from asyncio.subprocess import create_subprocess_shell

async def run_cmd_from(cmd: str, path: str = '.'):
    # Wrapper for creating async cmd runner
    proc = await create_subprocess_shell(f"cd {path} && {cmd}", stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        print(f"Error running command '{cmd}': {stderr}")

def tqdm_async(total=None, desc=None):
    # Decorator for adding progress bar for async function
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            progress_bar = tqdm(total=total, desc=desc)

            async def report_progress(coro):
                result = await coro
                progress_bar.update(1)
                return result

            tasks = [report_progress(func(*args, **kwargs))]

            # Close the progress bar once all tasks are completed
            await asyncio.gather(*tasks)
            progress_bar.close()

        return wrapper
    return decorator

def async_timer(total=None):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
            if total:
                remaining_time = max(total - execution_time, 0)
                print(f"Remaining time: {remaining_time:.4f} seconds")
            return result
        return wrapper
    return decorator