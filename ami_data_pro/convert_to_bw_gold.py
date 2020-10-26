# -*- coding:utf-8 -*-
"""
notes:
    1. 
"""
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function


import sys


if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    n_top = int(sys.argv[3])

    with open(in_path, encoding='utf-8') as sf, open(out_path, 'wt', encoding='utf-8') as of:
        for line in sf:
            line = line.strip()
            for _ in range(n_top):
                of.write('{}\n'.format(line))
            of.flush()






