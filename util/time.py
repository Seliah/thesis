from datetime import datetime

# This is 0000-00-00...
# See https://stackoverflow.com/questions/35851104/how-to-initialize-datetime-0000-00-00-000000-in-python
TIME_ZERO = datetime.min.time()


def seconds_since_midnight():
    """See https://stackoverflow.com/questions/15971308/get-seconds-since-midnight-in-python"""
    now = datetime.now()
    midnight = datetime.combine(now.date(), TIME_ZERO)
    return int((now - midnight).seconds)
