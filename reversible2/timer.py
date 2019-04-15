# https://gist.github.com/mauricioquiros/3243766
from timeit import default_timer


class Timer(object):
    def __init__(self, verbose=True, name=''):
        self.verbose = verbose
        self.timer = default_timer
        self.name = name

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs
        if self.verbose:
            print(
                'elapsed time {:s}: {:.3f} ms'.format(self.name, self.elapsed))