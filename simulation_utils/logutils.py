"""
Log utils borrowed from Dr. Michael Ekstrand
"""

import sys
import logging

_simple_format = logging.Formatter('{levelname} {name} {message}', style='{')


def setup(debug=False):
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(_simple_format)

    root = logging.getLogger()
    root.addHandler(ch)
    root.setLevel(logging.INFO)

    logging.getLogger('lenskit').setLevel(logging.DEBUG)
    logging.getLogger('simulation_utils').setLevel(logging.DEBUG)
    root.debug('log system configured')


def _logfile(file):
    fh = logging.FileHandler(file, mode='w')
    fmt = logging.Formatter(
        '{asctime} ({process}) [{levelname:7}] {name}: {message}', style='{'
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    return fh


def set_logfile(file):
    fh = _logfile(file)
    logging.getLogger().addHandler(fh)


class LogFile:
    """Context manager to add a log file to the system."""

    def __init__(self, file):
        self.file = file
        self.fh = None

    def __enter__(self):
        self.fh = _logfile(self.file)
        logging.getLogger().addHandler(self.fh)
        logging.getLogger(__name__).info('activated log file %s', self.file)

    def __exit__(self, *args, **kwargs):
        logging.getLogger(__name__).info('deactivating log file %s', self.file)
        logging.getLogger().removeHandler(self.fh)
