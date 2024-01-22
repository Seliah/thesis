"""Module for defining common time functions."""
from datetime import datetime

# This is 0000-00-00...
# See https://stackoverflow.com/questions/35851104/how-to-initialize-datetime-0000-00-00-000000-in-python
TIME_ZERO = datetime.min.time()


def get_date_floored(time: datetime):
    return datetime.combine(time.date(), TIME_ZERO)


def today():
    return get_date_floored(datetime.now())


def seconds_since_midnight(time: datetime):
    """See https://stackoverflow.com/questions/15971308/get-seconds-since-midnight-in-python."""
    return int((time - get_date_floored(time)).seconds)
