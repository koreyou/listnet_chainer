
from collections import defaultdict
import json
from tempfile import NamedTemporaryFile
import os


class Reporter(object):
    def __init__(self, path):
        # log is indexed by the key.
        self.log = defaultdict(lambda: defaultdict(dict))
        self.path = path

    def add_record(self, key, value, global_step, group='root'):
        self.log[group][str(key)][global_step] = float(value)

    def save(self):
        save_dir = os.path.dirname(self.path)
        prefix = os.path.basename(self.path) + '.'
        with NamedTemporaryFile('w', delete=False, dir=save_dir, prefix=prefix) as fout:
            json.dump(self.log, fout, indent=2)
            tempname = fout.name
        os.rename(tempname, self.path)

    def to_timeline(self):
        """ Aggregate records and crete timeline data that is
        suitable for drawing graph.

        Returned dict is indexed by data label. Each value
        is a tuple whose first item is the global step and second
        item is the values.

        Returns:
            dict
        """
        timeline = defaultdict(dict)
        for gl, group in self.log.iteritems():
            for l, data in group.iteritems():
                timeline[gl][l] = ([], [])
                for t, v in sorted(data.iteritems(), key=lambda x: x[0]):
                    timeline[gl][l][0].append(t)
                    timeline[gl][l][1].append(v)
        return timeline

    @classmethod
    def load(cls, path):
        obj = cls(path)
        with open(obj.path, 'r') as fin:
            log = json.load(fin)

        # json key is always str
        for gl, group in log.iteritems():
            for l, timeline in group.iteritems():
                for t, v in timeline.iteritems():
                    obj.log[gl][l][int(t)] = float(v)
        return obj
