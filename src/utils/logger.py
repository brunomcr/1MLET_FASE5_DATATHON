import logging
import sys

def setup_logger():
    # Configure logger
    logger = logging.getLogger('ETLLogger')
    logger.setLevel(logging.INFO)

    # Create stdout handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    # Define log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger

# Create global logger instance
logger = setup_logger() 