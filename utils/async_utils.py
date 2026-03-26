import asyncio
import inspect
from functools import wraps
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import torch
import asyncio
from typing import Any, Optional
import os
import multiprocessing
import traceback
from utils.logger import logger
import threading


class DebugEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    def __init__(self, slow_ms):
        super().__init__()
        if slow_ms > 0:
            self.slow_ms = slow_ms / 1000
        else:
            self.slow_ms = None

    def new_event_loop(self):
        loop = super().new_event_loop()

        if self.slow_ms is not None:
            # Enable debug features
            loop.set_debug(True)
            loop.slow_callback_duration = self.slow_ms
            logger.debug(f"\033[31m[============Patch {self.slow_ms=}============]\033[0m")

        # Store original task factory
        original_task_factory = loop.get_task_factory()

        def task_factory(loop, coro):
            # Get the original function name
            if hasattr(coro, "__qualname__"):
                func_name = coro.__qualname__
            elif hasattr(coro, "__name__"):
                func_name = coro.__name__
            else:
                func_name = str(coro)

            # Create the task using original factory if it exists
            if original_task_factory:
                task = original_task_factory(loop, coro)
            else:
                task = asyncio.tasks.Task(coro, loop=loop)

            # Add our custom attribute
            task._mt_raw_func_name = func_name
            return task

        logger.debug(f"\033[31m[============Patch new_event_loop add _mt_raw_func_name============]\033[0m")

        loop.set_task_factory(task_factory)
        return loop


def monkey_patch_event_loop(slow_ms=-1):
    asyncio.set_event_loop_policy(DebugEventLoopPolicy(slow_ms))


def get_coro_location(coro):
    try:
        frame = coro.cr_frame
        frame_info = inspect.getframeinfo(frame)
        return frame_info.filename, frame_info.lineno
    except:
        return "unknown", -1


def get_active_tasks(loop=None):
    if loop is None:
        loop = asyncio.get_running_loop()

    active_tasks = []
    for task in asyncio.all_tasks(loop):
        if hasattr(task, "_mt_raw_func_name"):
            task_name = task._mt_raw_func_name
        else:
            task_name = str(task)

        coro = task.get_coro()
        if hasattr(coro, "_mt_msg"):
            task_name += f"[{coro._mt_msg}]"

        line_no = get_coro_location(coro)[1]
        if line_no >= 0:
            s = f"{task_name}:{line_no}"
        else:
            s = task_name

        active_tasks.append(s)

    return active_tasks


def wrap_async_func(async_func, msg=None):
    @wraps(async_func)
    async def wrapped(*args, **kwargs):
        return await async_func(*args, **kwargs)

    wrapped._mt_msg = msg
    return wrapped


async def aiter(iterable):
    for item in iterable:
        yield item


def _dummy_task():
    return True  # Just for forcing process initialization


def worker_init(rank, init_fn, args, kwargs):
    logger.info(f"\033[32m[worker_init][{rank=}, {init_fn=}, {args=}, {kwargs=}]\033[0m")
    # Get unique CUDA rank for this worker
    with rank.get_lock():
        cur_rank = rank.value
        rank.value += 1

    torch.cuda.set_device(cur_rank)

    if init_fn:
        try:
            init_fn(*args, **kwargs)
        except:
            logger.error(f"\033[31m[[worker_init Error] {traceback.format_exc()}]\033[0m")
            raise RuntimeError("")

    logger.info(f"\033[32m[Worker {os.getpid()} initialized with CUDA {cur_rank}]\033[0m")


def init_process_pool(init_fn=None, num_workers=None, args=(), kwargs=None):
    kwargs = kwargs or {}
    ctx = multiprocessing.get_context("spawn")
    rank = ctx.Value("i", 0, lock=True)

    initargs = (rank, init_fn, args, kwargs)

    pool = ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=worker_init,
        initargs=initargs,
    )

    try:
        # Force all workers to initialize
        futures = [pool.submit(_dummy_task) for _ in range(num_workers)]
        for f in futures:
            f.result()  # Blocks until all workers are initialized
    except:
        logger.error(f"\033[31m[[init_process_pool Error] {traceback.format_exc()}]\033[0m")
        raise RuntimeError("")

    return pool


async def async_run_in_process(fn, process_pool, args=(), kwargs={}):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(process_pool, partial(fn, *args, **kwargs))


def sync_run_in_process(fn, process_pool, *args, **kwargs):
    future = process_pool.submit(partial(fn, *args, **kwargs))
    return future.result()  # This will block until complete


_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_loop_lock = threading.Lock()


def _create_background_loop():
    global _loop, _loop_thread
    if _loop is None:
        _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(target=_loop.run_forever, daemon=True, name="EventLoopThread")
        _loop_thread.start()
        logger.info("\033[35m[Created new event loop in background thread]\033[0m")


def _run_coroutine_in_background(coro):
    with _loop_lock:
        _create_background_loop()
        future = asyncio.run_coroutine_threadsafe(coro, _loop)
        try:
            return future.result()
        except Exception as e:
            logger.error(f"\033[31m[Coroutine failed: {traceback.format_exc()}]\033[0m")
            raise


def sync_run_async(coro) -> Any:
    global _loop, _loop_thread

    try:
        # 尝试获取当前线程的 running loop
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # 如果当前线程的 loop 正在运行，则在后台线程中运行协程
            return _run_coroutine_in_background(coro)
        else:
            # 如果当前线程的 loop 没有运行，直接运行
            return loop.run_until_complete(coro)
    except RuntimeError:
        # 没有 running loop，在后台线程中运行协程
        return _run_coroutine_in_background(coro)


def async_gen_to_list(async_gen) -> list:
    async def _collect():
        return [item async for item in async_gen]

    return sync_run_async(_collect())
