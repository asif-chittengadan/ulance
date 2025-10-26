import cv2
import asyncio
import logging
from backend.processors import UlancProcessor, NormalProcessor

class VideoStreamer:
    def __init__(self, shared_state, executor):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam. Make sure it is not in use by another application.")
        
        self.ulanc_processor = UlancProcessor()
        self.normal_processor = NormalProcessor()
        
        self.ulanc_frame_queue = asyncio.Queue(maxsize=1)
        self.normal_frame_queue = asyncio.Queue(maxsize=1)
        
        self.shared_state = shared_state
        self.executor = executor
        self.is_running = False
        logging.info("VideoStreamer initialized for threaded processing.")

    def _capture_and_process_sync(self):
        """
        A synchronous, blocking function that performs one full processing cycle.
        This is designed to be run in a thread pool executor.
        """
        ret, frame = self.cap.read()
        if not ret:
            # This log is important for debugging in the thread
            logging.warning("Failed to grab frame from webcam in worker thread.")
            return None, None

        current_quality = self.shared_state.get('quality_level', 'high')
        
        # Process frames one after another in the same thread
        ulanc_frame = self.ulanc_processor.process_frame(frame, current_quality)
        normal_frame = self.normal_processor.process_frame(frame, current_quality)
        
        return ulanc_frame, normal_frame

    async def capture_and_process_worker(self):
        self.is_running = True
        loop = asyncio.get_running_loop()
        logging.info("Threaded video processing worker started.")
        
        while self.is_running:
            try:
                # Run the blocking capture/process function in the executor
                ulanc_frame, normal_frame = await loop.run_in_executor(
                    self.executor, self._capture_and_process_sync
                )

                if ulanc_frame is None or normal_frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # Helper to update queue without blocking
                async def update_queue(queue, item):
                    if queue.full():
                        await queue.get() # Discard old frame if consumer is slow
                    await queue.put(item)

                # Put the results back into the async queues for the server to use
                await asyncio.gather(
                    update_queue(self.ulanc_frame_queue, ulanc_frame),
                    update_queue(self.normal_frame_queue, normal_frame)
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in capture_and_process_worker loop: {e}")
                await asyncio.sleep(1)

    async def get_jpeg_frame(self, stream_type):
        if stream_type == 'ulanc':
            frame = await self.ulanc_frame_queue.get()
        else: # normal
            frame = await self.normal_frame_queue.get()
            
        if frame is None:
            return None
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return jpeg.tobytes()

    def stop(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        logging.info("VideoStreamer stopped.")