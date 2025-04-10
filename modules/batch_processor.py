import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Coroutine
from enum import Enum
from concurrent.futures import CancelledError
import uuid # For unique batch IDs

# Import config for logging level
import config

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"

# Type hint for the processing function
ProcessItemFunc = Callable[[Any, Optional[Any]], Coroutine[Any, Any, Dict[str, Any]]]
# Progress callback type: (processed_count, total_count, item_id, item_result)
ProgressCallback = Callable[[int, int, Any, Optional[Dict[str, Any]]], Coroutine[Any, Any, None]]

class BatchProcessor:
    """
    Manages the execution of batch processing tasks asynchronously.
    Iterates over items and calls a provided processing function for each.
    Handles concurrency, progress tracking, and control signals (stop, pause).
    """

    def __init__(self, max_concurrent_tasks: int = 5):
        """
        Initializes the BatchProcessor.

        Args:
            max_concurrent_tasks: Maximum number of items to process concurrently.
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self._current_batch_id: Optional[str] = None
        self._current_task: Optional[asyncio.Task] = None
        self._items_to_process: List[Any] = []
        self._process_item_func: Optional[ProcessItemFunc] = None
        self._progress_callback: Optional[ProgressCallback] = None
        self._state_manager: Optional[Any] = None # Optional state manager for detailed updates

        # Internal state
        self._status: BatchStatus = BatchStatus.IDLE
        self._processed_count: int = 0
        self._total_count: int = 0
        self._results: Dict[Any, Dict[str, Any]] = {} # Store results keyed by item identifier
        self._errors: Dict[Any, str] = {}
        self._stop_requested: bool = False
        self._pause_requested: bool = False
        self._semaphore: Optional[asyncio.Semaphore] = None

    @property
    def status(self) -> BatchStatus:
        return self._status

    def get_progress(self) -> Dict[str, Any]:
        """Returns the current progress of the batch."""
        return {
            "batch_id": self._current_batch_id,
            "status": self._status.value,
            "total_items": self._total_count,
            "processed_items": self._processed_count,
            "success_count": len(self._results),
            "error_count": len(self._errors),
            # Optionally include results/errors if needed, but can get large
            # "results": self._results,
            # "errors": self._errors
        }

    async def start_batch(
        self,
        items: List[Any],
        item_processor: ProcessItemFunc,
        progress_callback: Optional[ProgressCallback] = None,
        state_manager: Optional[Any] = None,
        item_id_func: Optional[Callable[[Any], Any]] = None # Function to get a unique ID from an item
    ) -> str:
        """
        Starts a new batch processing job.

        Args:
            items: The list of items to process.
            item_processor: The async function to call for each item.
                           It should accept the item and optionally the state_manager.
                           It must return a dictionary representing the result.
            progress_callback: An optional async function called after each item is processed.
                               Receives (processed_count, total_count, item_id, item_result).
            state_manager: An optional state manager object to pass to the item_processor
                           and potentially use for internal status updates.
            item_id_func: An optional function to extract a unique, hashable identifier from each item
                          (e.g., lambda keyword: keyword.name). Defaults to using the item itself.

        Returns:
            The unique ID of the started batch.

        Raises:
            RuntimeError: If a batch is already running.
        """
        if self._status == BatchStatus.RUNNING or self._status == BatchStatus.PAUSED:
            raise RuntimeError(f"Cannot start a new batch while one is {self._status.value}. Batch ID: {self._current_batch_id}")

        self._current_batch_id = str(uuid.uuid4())
        logger.info(f"Starting new batch (ID: {self._current_batch_id}) with {len(items)} items.")

        # Reset state
        self._items_to_process = list(items)
        self._process_item_func = item_processor
        self._progress_callback = progress_callback
        self._state_manager = state_manager
        self._item_id_func = item_id_func or (lambda x: x) # Default to using item itself as ID

        self._status = BatchStatus.RUNNING
        self._processed_count = 0
        self._total_count = len(items)
        self._results = {}
        self._errors = {}
        self._stop_requested = False
        self._pause_requested = False
        self._semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        # Start the background task
        self._current_task = asyncio.create_task(self._run_batch())
        self._current_task.add_done_callback(self._batch_finished_callback)

        return self._current_batch_id

    async def stop_batch(self):
        """Requests the current batch to stop gracefully."""
        if self._status not in [BatchStatus.RUNNING, BatchStatus.PAUSED]:
            logger.warning(f"No batch running or paused to stop (Status: {self._status.value}).")
            return

        logger.info(f"Requesting stop for batch ID: {self._current_batch_id}")
        self._stop_requested = True
        self._status = BatchStatus.STOPPING
        if self._pause_requested: # If paused, resume briefly to allow cancellation check
             self._pause_requested = False

        if self._current_task:
            # Give the task some time to notice the stop request
            await asyncio.sleep(0.1)
            # Optionally cancel if it doesn't stop quickly (might leave tasks running)
            # self._current_task.cancel()

    async def pause_batch(self):
        """Requests the current batch to pause."""
        if self._status != BatchStatus.RUNNING:
            logger.warning(f"Batch is not running, cannot pause (Status: {self._status.value}).")
            return
        logger.info(f"Requesting pause for batch ID: {self._current_batch_id}")
        self._pause_requested = True

    async def resume_batch(self):
        """Resumes a paused batch."""
        if self._status != BatchStatus.PAUSED:
            logger.warning(f"Batch is not paused, cannot resume (Status: {self._status.value}).")
            return
        logger.info(f"Resuming batch ID: {self._current_batch_id}")
        self._pause_requested = False
        # The running task will detect this change in its loop

    async def _run_batch(self):
        """The main asynchronous loop for processing batch items."""
        logger.info(f"Batch {self._current_batch_id} execution started.")
        tasks = []
        processed_indices = set()

        try:
            for index, item in enumerate(self._items_to_process):
                if self._stop_requested:
                    logger.info(f"Stop requested detected in batch {self._current_batch_id}. Breaking loop.")
                    self._status = BatchStatus.CANCELLED
                    break

                while self._pause_requested:
                    if self._status != BatchStatus.PAUSED:
                         logger.info(f"Batch {self._current_batch_id} paused.")
                         self._status = BatchStatus.PAUSED
                    await asyncio.sleep(0.5) # Check pause/stop status periodically
                    if self._stop_requested: # Check again after sleep
                         logger.info(f"Stop requested detected while paused in batch {self._current_batch_id}.")
                         self._status = BatchStatus.CANCELLED
                         # Need to break outer loop as well
                         raise asyncio.CancelledError("Batch cancelled while paused")

                if self._status == BatchStatus.PAUSED: # If resuming
                     logger.info(f"Batch {self._current_batch_id} resumed.")
                     self._status = BatchStatus.RUNNING

                # Acquire semaphore before creating task
                await self._semaphore.acquire()

                # Create and schedule the task for the current item
                task = asyncio.create_task(self._process_single_item(item, index))
                tasks.append(task)

                # Process completed tasks to release semaphore and handle results
                if len(tasks) >= self.max_concurrent_tasks or index == self._total_count - 1:
                    # Wait for at least one task to complete
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = list(pending) # Update list of running tasks
                    # Process results of completed tasks
                    await self._handle_completed_tasks(done, processed_indices)


            # Wait for any remaining tasks to complete after the loop
            if tasks:
                 logger.info(f"Waiting for {len(tasks)} remaining tasks to complete...")
                 done, _ = await asyncio.wait(tasks)
                 await self._handle_completed_tasks(done, processed_indices)

            if self._status == BatchStatus.RUNNING: # If not cancelled or errored
                 self._status = BatchStatus.COMPLETED
                 logger.info(f"Batch {self._current_batch_id} completed successfully.")

        except asyncio.CancelledError:
             logger.info(f"Batch {self._current_batch_id} was cancelled.")
             self._status = BatchStatus.CANCELLED
             # Cancel any remaining running tasks
             for t in tasks:
                 t.cancel()
             if tasks:
                 await asyncio.wait(tasks) # Wait for cancellations to process
        except Exception as e:
            logger.error(f"Critical error during batch {self._current_batch_id} execution: {e}", exc_info=True)
            self._status = BatchStatus.ERROR
            # Cancel remaining tasks on critical error
            for t in tasks:
                 t.cancel()
            if tasks:
                 await asyncio.wait(tasks)
        finally:
            logger.info(f"Batch {self._current_batch_id} finished with status: {self._status.value}")


    async def _process_single_item(self, item: Any, index: int) -> tuple[int, Any, Optional[Dict[str, Any]], Optional[str]]:
        """Wrapper to process a single item and handle exceptions."""
        item_id = None
        try:
            item_id = self._item_id_func(item) # Get unique ID for results dict
            logger.debug(f"Processing item {index+1}/{self._total_count} (ID: {item_id})...")
            # Pass state_manager if the processor function accepts it
            # This requires introspection or assuming the function signature
            # Simpler: Assume it takes (item, state_manager)
            if self._process_item_func:
                 result_data = await self._process_item_func(item, self._state_manager)
                 logger.debug(f"Successfully processed item {index+1} (ID: {item_id}).")
                 return index, item_id, result_data, None # index, id, result, error
            else:
                 raise RuntimeError("process_item_func not set")
        except Exception as e:
            logger.error(f"Error processing item {index+1} (ID: {item_id}): {e}", exc_info=True)
            return index, item_id, None, str(e) # index, id, result, error
        finally:
             # Release semaphore after task completion or failure
             if self._semaphore:
                 self._semaphore.release()


    async def _handle_completed_tasks(self, done_tasks: set[asyncio.Task], processed_indices: set[int]):
         """Processes results from completed tasks."""
         for task in done_tasks:
            try:
                index, item_id, result_data, error_msg = task.result()

                if index in processed_indices: # Avoid double processing if task somehow completes twice
                     continue
                processed_indices.add(index)

                self._processed_count += 1
                item_result_for_callback = None

                if error_msg:
                    self._errors[item_id] = error_msg
                    item_result_for_callback = {"success": False, "error": error_msg}
                elif result_data is not None:
                    self._results[item_id] = result_data
                    item_result_for_callback = result_data # Assumes result_data includes success status
                    if "success" not in item_result_for_callback:
                         item_result_for_callback["success"] = True # Assume success if no error and result exists

                # Call progress callback
                if self._progress_callback:
                    try:
                        await self._progress_callback(
                            self._processed_count,
                            self._total_count,
                            item_id,
                            item_result_for_callback
                        )
                    except Exception as cb_error:
                        logger.error(f"Error in progress callback for item {item_id}: {cb_error}")

            except asyncio.CancelledError:
                 logger.warning(f"A processing task for batch {self._current_batch_id} was cancelled.")
                 # Don't increment processed count for cancelled tasks
            except Exception as e:
                 logger.error(f"Error retrieving result from task: {e}", exc_info=True)
                 # How to handle this? Maybe mark an unknown item as errored?
                 self._processed_count += 1 # Count it as processed, but with an error
                 unknown_item_id = f"unknown_item_{self._processed_count}"
                 self._errors[unknown_item_id] = f"Failed to retrieve task result: {e}"


    def _batch_finished_callback(self, task: asyncio.Task):
        """Callback executed when the main batch task finishes."""
        try:
            # Check if the task raised an exception
            exc = task.exception()
            if exc and not isinstance(exc, asyncio.CancelledError):
                logger.error(f"Batch task {self._current_batch_id} finished with unhandled exception: {exc}", exc_info=exc)
                if self._status != BatchStatus.CANCELLED: # Don't override cancelled status
                     self._status = BatchStatus.ERROR
            elif task.cancelled():
                 logger.info(f"Batch task {self._current_batch_id} was cancelled.")
                 self._status = BatchStatus.CANCELLED # Ensure status is set if cancelled externally
            else:
                 # If status is still RUNNING, mark as COMPLETED
                 if self._status == BatchStatus.RUNNING:
                      self._status = BatchStatus.COMPLETED
                 logger.info(f"Batch task {self._current_batch_id} finished naturally with status {self._status.value}.")

        except asyncio.CancelledError:
             logger.info(f"Batch task {self._current_batch_id} finished callback cancelled.")
             self._status = BatchStatus.CANCELLED
        except Exception as e:
            logger.error(f"Error in _batch_finished_callback for batch {self._current_batch_id}: {e}", exc_info=True)
            if self._status not in [BatchStatus.CANCELLED, BatchStatus.COMPLETED]:
                 self._status = BatchStatus.ERROR
        finally:
             # Clean up references
             self._current_task = None
             # Keep batch ID and results until a new batch starts


# Example Usage
if __name__ == "__main__":

    # Example item processing function (async)
    async def example_process_item(item: Dict[str, Any], state_manager: Optional[Any]) -> Dict[str, Any]:
        item_id = item.get("id", "unknown")
        duration = random.uniform(0.5, 2.0)
        logger.info(f"Processing item {item_id} (will take {duration:.1f}s)...")
        await asyncio.sleep(duration)
        # Simulate potential failure
        if random.random() < 0.1:
            logger.error(f"Simulated failure for item {item_id}")
            raise ValueError(f"Simulated processing error for {item_id}")
        logger.info(f"Finished processing item {item_id}")
        return {"success": True, "item_id": item_id, "processed_at": time.time(), "duration": duration}

    # Example progress callback (async)
    async def example_progress_callback(processed, total, item_id, result):
        status = "Success" if result and result.get("success") else "Failed"
        error_msg = f" Error: {result.get('error')}" if result and not result.get("success") else ""
        print(f"Progress: {processed}/{total} | Item ID: {item_id} | Status: {status}{error_msg}")
        await asyncio.sleep(0.01) # Simulate async work in callback

    async def run_example():
        print("Testing BatchProcessor...")
        processor = BatchProcessor(max_concurrent_tasks=3)

        items_to_process = [{"id": i, "data": f"data_{i}"} for i in range(10)]

        print(f"Initial Status: {processor.status.value}")

        try:
            batch_id = await processor.start_batch(
                items=items_to_process,
                item_processor=example_process_item,
                progress_callback=example_progress_callback,
                item_id_func=lambda item: item["id"] # Use 'id' field as the key
            )
            print(f"Batch started with ID: {batch_id}")
            print(f"Status after start: {processor.status.value}")

            # Let the batch run for a while
            start_time = time.time()
            while processor.status == BatchStatus.RUNNING:
                 # Simulate pausing after a few seconds
                 if time.time() - start_time > 4 and processor.status == BatchStatus.RUNNING:
                      print("\n--- Requesting Pause ---")
                      await processor.pause_batch()
                      print(f"Status after pause request: {processor.status.value}")
                      await asyncio.sleep(2) # Stay paused
                      print("--- Requesting Resume ---")
                      await processor.resume_batch()
                      print(f"Status after resume request: {processor.status.value}")
                      start_time = time.time() # Reset timer to avoid re-pausing immediately

                 # Simulate stopping after some more time
                 # if time.time() - start_time > 8 and processor.status == BatchStatus.RUNNING:
                 #      print("\n--- Requesting Stop ---")
                 #      await processor.stop_batch()
                 #      print(f"Status after stop request: {processor.status.value}")

                 progress = processor.get_progress()
                 print(f"  -> Running... Processed: {progress['processed_items']}/{progress['total_items']} (Errors: {progress['error_count']})")
                 await asyncio.sleep(1)

            # Wait for the batch task to fully complete (if not stopped)
            if processor._current_task:
                 await processor._current_task

            print(f"\nBatch finished with status: {processor.status.value}")
            final_progress = processor.get_progress()
            print("Final Progress:")
            print(json.dumps(final_progress, indent=2))

            # Access results/errors (example)
            # print("\nResults:")
            # print(json.dumps(processor._results, indent=2))
            # print("\nErrors:")
            # print(json.dumps(processor._errors, indent=2))


        except RuntimeError as e:
            print(f"Runtime Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(run_example())
