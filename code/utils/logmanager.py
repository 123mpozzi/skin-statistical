import logging
import datetime
import os


# Credits to
# https://gist.github.com/huklee/cea20761dd05da7c39120084f52fcc7c?permalink_comment_id=3976951#gistcomment-3976951
class Logger:
    _logger = None

    def __new__(cls, *args, **kwargs):
        if cls._logger is None:

            print("Logger new")
            cls._logger = super().__new__(cls, *args, **kwargs)
            cls._logger = logging.getLogger("skin")
            cls._logger.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s')

            now = datetime.datetime.now()
            dirname = os.path.join('..', 'logs')

            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            log_filename = "log_" + now.strftime("%Y-%m-%d") + ".log"
            fileHandler = logging.FileHandler(os.path.join(dirname, log_filename))

            streamHandler = logging.StreamHandler()

            fileHandler.setFormatter(formatter)
            streamHandler.setFormatter(formatter)

            cls._logger.addHandler(fileHandler)
            cls._logger.addHandler(streamHandler)

        return cls._logger


logger = Logger()

def info(msg: str):
    logger.info(msg)

def debug(msg: str):
    logger.debug(msg)

def warning(msg: str):
    logger.warning(msg)

def error(msg: str):
    logger.error(msg)

def critical(msg: str):
    logger.critical(msg)

