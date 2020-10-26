# -*- coding:utf-8 -*-
import sys
import json


if __name__ == '__main__':
    sf = sys.argv[1]
    of_in = sys.argv[2]
    of_out = sys.argv[3]

    datas = []
    with open(sf, encoding='utf-8') as sf, \
            open(of_in, 'wt', encoding='utf-8') as oi, open(of_out, 'wt', encoding='utf-8') as oo:
        for line in sf:
            sample = json.loads(line)
            src = sample['src']
            idx = len(src) - 2
            while idx > -1:
                idx -= 1
                tok = src[idx]
                if tok in ('<a>', '<b>'):
                    break
            new_tgt = ' '.join(src[idx+1: -2])
            new_src_pre = ' '.join(src[:idx+1]) + ' <mask> ' + src[-2] + ' '
            top_preds = sample['top_preds']
            for pred in top_preds:
                pred_sent = ' '.join(pred['pred'])
                oi.write('{}\n'.format(new_src_pre + pred_sent))
                oi.flush()
                oo.write('{}\n'.format(new_tgt))
                oo.flush()




