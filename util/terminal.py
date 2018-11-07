import sys

import logging


class ProgressBar:
    def __init__(self, count=0, total=0, width=50):
        self.count = count
        self.total = total
        self.width = width
        self.flush_rate = max(width, 100)
        if total == 0:
            self.ratio = 0
        else:
            self.ratio = int(self.flush_rate * count / total)

    def _show(self, ratio):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        progress = int(self.width * ratio / self.flush_rate)
        percent = int(100 * ratio / self.flush_rate)
        sys.stdout.write('{0:3}%: '.format(percent))
        sys.stdout.write('#' * progress + '-' * (self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def log(self):
        self.count += 1
        tmp = int(self.flush_rate * self.count / self.total)
        if tmp > self.ratio:
            self._show(tmp)
            self.ratio = tmp


def setup_logging_config(debug=False):
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    if debug:
        logging.root.setLevel(level=logging.DEBUG)
    else:
        logging.root.setLevel(level=logging.INFO)
