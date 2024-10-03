
from datetime import datetime
import time


def sleep_minute(n_minute):
    time.sleep(60 * n_minute)


def sleep_sec(n_sec):
    time.sleep(n_sec)


class TimeUtils:
    def __init__(self):
        pass

    DDYYYYMM = "%d%m%Y"
    YYYY_MM_DD = "%Y-%m-%d"
    YYYYMMDD = "%Y%m%d"
    YYYYMMDDHHMMSS = "%Y-%m-%d %H:%M:%S"
    YYYYMMDDHHMMSS_COMPACT = "%Y%m%d_%H%M%S"
    YYYYMMDDHHMM_COMPACT = "%Y%m%d_%H%M"

    @staticmethod
    def get_now_str(fmt=YYYYMMDDHHMMSS_COMPACT):
        return datetime.today().strftime(fmt)


class TimeAccumulator(object):
    def __init__(self):
        self.total_time = 0  # ms

    def add(self, tc):
        self.total_time += tc

    def get_total_time(self):
        # return sec
        return int(self.total_time / 1000)


class TimeCounter(object):
    def __init__(self):
        self.start = time.time()
        self.total_eval_time = 0  # sec

    def restart(self):
        self.start = time.time()

    def add_eval_time(self, eval_time):
        self.total_eval_time = self.total_eval_time + eval_time

    def count(self):
        return int(time.time() - self.start) - self.total_eval_time

    def count_ms(self):
        ts = time.time() - self.start
        return int(ts * 1000 - self.total_eval_time*1000)

