import time
import signal
import sys
from loguru import logger
from sentinel_system import SentinelSystem

def main():
    system = SentinelSystem()
    
    # Setup signal handlers
    def shutdown(signum, frame):
        logger.info("Shutdown signal received...")
        system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    system.start()
    
    try:
        while True:
            result = system.process_step()
            if result is None:
                time.sleep(0.01)
                continue
            
            # CLI output is already handled by logger inside process_step
            # We can add more specific logic here if needed

    except Exception as e:
        logger.exception("Unexpected error in main loop")
    finally:
        system.stop()
        logger.info("System shutdown complete.")

if __name__ == "__main__":
    main()
