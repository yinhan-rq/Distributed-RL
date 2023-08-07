# -*- coding: utf-8 -*-
from hashlib import new
import logging
import logging.handlers
import os
import time

__all__ = ["LogDebug", "LogErr", "LogExc"]


log_path = os.path.dirname(os.path.abspath(__file__)) + "/logs"
print(log_path)

debug_log_path = log_path + "/debug.log"
error_log_path = log_path + "/error.log"

class MyTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    def doRollover(self):
        if self.stream:
            self.stream.close()
        t = self.rolloverAt - self.interval
        
        timeTuple = time.gmtime(t) if self.utc else time.localtime(t)
        dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)
        if not os.path.exists(dfn):
            os.rename(self.baseFilename, dfn)
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        self.mode = 'a'
        self.stream = self._open()
        currentTime = int(time.time())
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstNow = time.localtime(currentTime)[-1]
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:
                    newRolloverAt = newRolloverAt - 3600
                else:
                    newRolloverAt = newRolloverAt + 3600
        self.rolloverAt = newRolloverAt


def Log(name, level, path):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="[%(levelname)1.1s %(asctime)s.%(msecs)03d %(process)d:%(processName)s %(filename)s:%(funcName)s:%(lineno)s] - %(message)s ",
        datefmt='%Y%m%d %H:%M:%S')
    channel = MyTimedRotatingFileHandler(filename=path, when="MIDNIGHT", interval=1, backupCount=3)
    channel.setFormatter(formatter)
    logger.addHandler(channel)
    #logger = logging.LoggerAdapter(logger, {})
    return logger

LogDebug = Log("debug", logging.DEBUG, debug_log_path).debug
LogErr = Log("error", logging.ERROR, error_log_path).error
LogExc = Log("exception", logging.ERROR, error_log_path).exception