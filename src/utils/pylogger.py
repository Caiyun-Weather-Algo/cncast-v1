import logging
from logging.handlers import RotatingFileHandler
from typing import Mapping, Optional
from datetime import datetime
import os


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
        log_dir: str = "/path/to/log",
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        :param log_dir: The directory to save logs to. Default is "/path/to/log".
        """
        logger = logging.getLogger(name)
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        # Set up rotating file handler (500MB = 500 * 1024 * 1024 bytes)
        handler = RotatingFileHandler(
            filename=f"{log_dir}/log.txt",
            maxBytes=500 * 1024 * 1024,  # 500MB
            backupCount=5  # Keep up to 5 backup files (log1.txt, log2.txt etc)
        )
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(
        self, msg: str, level: int = 20, rank: Optional[int] = None, *args, **kwargs
    ) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
            CRITICAL = 50
            FATAL = CRITICAL
            ERROR = 40
            WARNING = 30
            WARN = WARNING
            INFO = 20
            DEBUG = 10
            NOTSET = 0
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            msg = f"{timestamp}: {msg}"
            msg, kwargs = self.process(msg, kwargs)
            if self.rank_zero_only:
                if rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)
