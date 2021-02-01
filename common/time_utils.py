from datetime import datetime
from pytz import timezone

tz = timezone("US/Eastern")

def get_local_datetime(timestamp):
    return tz.fromutc(datetime.utcfromtimestamp(timestamp))

def get_local_unixtime(datetime):
    return datetime.replace(tzinfo=tz).timestamp()