import json
import os

from six.moves import cPickle


here = os.path.dirname(__file__)

class FilenameMapper:
    def __init__(self, cache_filename):
        if os.path.exists(cache_filename):
            with open(cache_filename, 'rb') as f:
                self.old2new, self.new2old = cPickle.load(f)
        else:
            self.old2new, self.new2old = FilenameMapper.create_filename_map(cache_filename)

    @staticmethod
    def create_filename_map(cache_filename):
        list_filenames = ['list_nosign.txt', 'list_nosign_s.txt', 'list_sign.txt', 'list_sign_s.txt']
        json_filenames = ['list_roadblock.json']
        s0 = set()
        s1 = set()
        for filename in list_filenames:
            with open(os.path.join(here, filename), 'r') as f:
                for line in f:
                    s0.add(line.strip().split('.')[0])
        for filename in json_filenames:
            with open(os.path.join(here, filename), 'r') as f:
                l = json.load(f)
            for filename in l:
                basename = filename.split('.')[0]
                s1.add(basename)
        v0 = sorted(list(s0))
        v1 = sorted(list(s1))
        old2new = dict()
        new2old = dict()
        for i, name_old in enumerate(v0):
            name_new = '%06d' % (i + 1)
            old2new[name_old] = name_new
            new2old[name_new] = name_old
        off1 = ((len(v0) - 1) // 10000 + 1) * 10000
        for i, name_old in enumerate(v1):
            name_new = '%06d' % (i + 1 + off1)
            old2new[name_old] = name_new
            new2old[name_new] = name_old
        rst = (old2new, new2old)
        with open(cache_filename, 'wb') as f:
            cPickle.dump(rst, f, protocol=2)
        return rst

def main():
    pass

if __name__ == "__main__":
    main()
