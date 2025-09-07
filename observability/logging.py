from __future__ import annotations
import logging
import sys
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, service_name: str = "docfoundry"):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info'
            }:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Create timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Format basic message
        message = f"{timestamp} | {record.levelname:8} | {record.name} | {record.getMessage()}"
        
        # Add colors if enabled
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            message = f"{color}{message}{self.RESET}"
        
        # Add exception info if present
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        
        return message

def setup_logging(
    level: str = "INFO",
    service_name: str = "docfoundry",
    log_file: Optional[str] = None,
    use_json: bool = False,
    use_colors: bool = True
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        service_name: Name of the service for structured logging
        log_file: Optional file path for file logging
        use_json: Whether to use JSON formatting
        use_colors: Whether to use colored output for console
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set root logger level
    root_logger.setLevel(numeric_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if use_json:
        console_formatter = JSONFormatter(service_name)
    else:
        console_formatter = ColoredFormatter(use_colors)
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        
        # Always use JSON for file logging
        file_formatter = JSONFormatter(service_name)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)

class StructuredLogger:
    """Wrapper for structured logging with additional context."""
    
    def __init__(self, name: str, **default_context):
        self.logger = logging.getLogger(name)
        self.default_context = default_context
    
    def _log(self, level: int, message: str, **context) -> None:
        """Log with structured context."""
        # Merge default context with provided context
        full_context = {**self.default_context, **context}
        
        # Create log record with extra fields
        extra = {f"ctx_{k}": v for k, v in full_context.items()}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **context) -> None:
        """Log debug message with context."""
        self._log(logging.DEBUG, message, **context)
    
    def info(self, message: str, **context) -> None:
        """Log info message with context."""
        self._log(logging.INFO, message, **context)
    
    def warning(self, message: str, **context) -> None:
        """Log warning message with context."""
        self._log(logging.WARNING, message, **context)
    
    def error(self, message: str, **context) -> None:
        """Log error message with context."""
        self._log(logging.ERROR, message, **context)
    
    def critical(self, message: str, **context) -> None:
        """Log critical message with context."""
        self._log(logging.CRITICAL, message, **context)
    
    def exception(self, message: str, **context) -> None:
        """Log exception with context."""
        full_context = {**self.default_context, **context}
        extra = {f"ctx_{k}": v for k, v in full_context.items()}
        self.logger.exception(message, extra=extra)

def get_structured_logger(name: str, **default_context) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name, **default_context)

# Logging decorators
def log_function_call(logger_name: Optional[str] = None, level: str = "DEBUG"):
    """Decorator to log function calls."""
    def decorator(func):
        logger = get_logger(logger_name or func.__module__)
        log_level = getattr(logging, level.upper(), logging.DEBUG)
        
        def wrapper(*args, **kwargs):
            logger.log(
                log_level,
                f"Calling {func.__name__}",
                extra={
                    "function": func.__name__,
                    "module": func.__module__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
            )
            
            try:
                result = func(*args, **kwargs)
                logger.log(
                    log_level,
                    f"Completed {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "module": func.__module__,
                        "status": "success"
                    }
                )
                return result
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {e}",
                    extra={
                        "function": func.__name__,
                        "module": func.__module__,
                        "error_type": type(e).__name__,
                        "status": "error"
                    },
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator

def log_performance(logger_name: Optional[str] = None, threshold_ms: float = 1000.0):
    """Decorator to log slow function performance."""
    def decorator(func):
        logger = get_logger(logger_name or func.__module__)
        
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                if duration_ms > threshold_ms:
                    logger.warning(
                        f"Slow function execution: {func.__name__}",
                        extra={
                            "function": func.__name__,
                            "module": func.__module__,
                            "duration_ms": duration_ms,
                            "threshold_ms": threshold_ms
                        }
                    )
                
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Function failed: {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "module": func.__module__,
                        "duration_ms": duration_ms,
                        "error_type": type(e).__name__
                    },
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator

# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(
        level="DEBUG",
        service_name="docfoundry-example",
        use_json=False,
        use_colors=True
    )
    
    # Regular logger
    logger = get_logger(__name__)
    logger.info("This is a regular log message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Structured logger
    struct_logger = get_structured_logger(
        __name__,
        component="example",
        version="1.0.0"
    )
    
    struct_logger.info(
        "Processing document",
        doc_id="example-123",
        doc_type="markdown",
        size_bytes=1024
    )
    
    # Function logging decorator
    @log_function_call(level="INFO")
    @log_performance(threshold_ms=100)
    def example_function(name: str, delay: float = 0.05):
        import time
        time.sleep(delay)
        return f"Hello, {name}!"
    
    # Test the decorated function
    result = example_function("World", delay=0.15)
    logger.info(f"Function result: {result}")
    
    # Test exception logging
    try:
        raise ValueError("Example error")
    except Exception:
        struct_logger.exception(
            "An error occurred during processing",
            operation="example",
            user_id="user123"
        )