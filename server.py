import asyncio
import logging
import json
import speedtest
from concurrent.futures import ThreadPoolExecutor
from aiohttp import web
from backend.video_streamer import VideoStreamer

logging.basicConfig(level=logging.INFO)

# --- Network Monitoring ---
def run_speed_test():
    """Synchronous function to run the speed test."""
    try:
        s = speedtest.Speedtest()
        s.get_best_server()
        s.upload(threads=1)
        upload_speed_mbps = s.results.upload / 1_000_000
        logging.info(f"Speed Test Result: {upload_speed_mbps:.2f} Mbps")
        return upload_speed_mbps
    except Exception as e:
        logging.error(f"Speed test failed: {e}")
        return 0.0

async def network_monitor_worker(app):
    """Asynchronous worker to periodically check network speed."""
    loop = asyncio.get_running_loop()
    logging.info("Network monitor started.")
    while True:
        try:
            # Run the blocking speedtest in a separate thread from the pool
            upload_speed = await loop.run_in_executor(app['executor'], run_speed_test)
            
            if upload_speed > 2.5:
                quality = 'high'
            elif 1.0 <= upload_speed <= 2.5:
                quality = 'medium'
            else:
                quality = 'low'
            
            app['shared_state']['upload_speed'] = f"{upload_speed:.2f}"
            app['shared_state']['quality_level'] = quality
            
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            logging.info("Network monitor stopping.")
            break
        except Exception as e:
            logging.error(f"Error in network monitor: {e}")
            await asyncio.sleep(30)

# --- Web Handlers ---
async def index_handler(request):
    return web.FileResponse('./frontend/index.html')

async def status_handler(request):
    state = request.app['shared_state']
    return web.json_response({
        'upload_speed': state.get('upload_speed', 'N/A'),
        'quality_level': state.get('quality_level', 'high')
    })

async def websocket_handler(request):
    stream_type = request.match_info.get('type')
    if stream_type not in ['ulanc', 'normal']:
        return web.HTTPNotFound()

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    request.app['websockets'][stream_type].add(ws)
    logging.info(f"Client connected to {stream_type} stream. Total clients: {len(request.app['websockets'][stream_type])}")

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.ERROR:
                logging.error(f'WS connection closed with exception {ws.exception()}')
    finally:
        request.app['websockets'][stream_type].remove(ws)
        logging.info(f"Client disconnected from {stream_type} stream. Remaining clients: {len(request.app['websockets'][stream_type])}")

    return ws

async def broadcast_frames(app):
    streamer = app['video_streamer']
    while True:
        try:
            ulanc_jpeg = await streamer.get_jpeg_frame('ulanc')
            normal_jpeg = await streamer.get_jpeg_frame('normal')

            await asyncio.gather(
                *[ws.send_bytes(ulanc_jpeg) for ws in app['websockets']['ulanc'] if not ws.closed],
                *[ws.send_bytes(normal_jpeg) for ws in app['websockets']['normal'] if not ws.closed]
            )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error(f"Error in broadcast loop: {e}")

# --- App Lifecycle ---
async def start_background_tasks(app):
    # Create a single thread pool for all blocking tasks
    app['executor'] = ThreadPoolExecutor(max_workers=2) # 1 for network, 1 for video

    app['shared_state'] = {'upload_speed': 'Testing...', 'quality_level': 'high'}
    app['video_streamer'] = VideoStreamer(app['shared_state'], app['executor'])
    
    app['video_worker'] = asyncio.create_task(app['video_streamer'].capture_and_process_worker())
    app['network_monitor'] = asyncio.create_task(network_monitor_worker(app))
    app['frame_broadcaster'] = asyncio.create_task(broadcast_frames(app))
    logging.info("All background tasks started.")

async def cleanup_background_tasks(app):
    logging.info("Cleaning up background tasks...")
    tasks = [app['network_monitor'], app['video_worker'], app['frame_broadcaster']]
    for task in tasks:
        task.cancel()
    app['video_streamer'].stop()
    await asyncio.gather(*tasks, return_exceptions=True)
    app['executor'].shutdown(wait=True)

def create_app():
    app = web.Application()
    app['websockets'] = {'ulanc': set(), 'normal': set()}
    
    app.router.add_get('/', index_handler)
    app.router.add_get('/status', status_handler)
    app.router.add_get('/ws/{type}', websocket_handler)
    
    app.on_startup.append(start_background_tasks)
    app.on_shutdown.append(cleanup_background_tasks)
    return app

if __name__ == '__main__':
    web.run_app(create_app(), host='0.0.0.0', port=8080)