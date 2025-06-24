from pathlib import Path
import logging

class Logger:
    """Centralized logging configuration"""
    
    @staticmethod
    def setup_logger(name: str, log_file: str = 'logs/classification_log.jsonl') -> logging.Logger:
        """Setup structured logging to file only"""
        
        # Create logs directory if it doesn't exist
        Path(log_file).parent.mkdir(exist_ok=True)
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # File handler for structured logs
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(message)s')  # JSON logs don't need timestamp prefix
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger