# -*- coding:utf-8 -*-
import sys
import json


if __name__ == '__main__':
    save_path = sys.argv[1]
    fw_raw_in = sys.argv[2]
    bw_raw_out = sys.argv[3]
    n_tops = int(sys.argv[4])

    bw_scores = []
    with open(bw_raw_out, encoding='utf-8') as bwf:
        for line in bwf:
            sample = json.loads(line)
            bw_scores.append(sample['gold_score'])

    idx = 0 - n_tops
    with open(fw_raw_in, encoding='utf-8') as fwf, \
            open(save_path, 'wt', encoding='utf-8') as sf:
        for line in fwf:
            idx += n_tops
            sample = json.loads(line)
            top_preds = sample['top_preds']
            n_tops = len(top_preds)
            scores = bw_scores[idx: idx + n_tops]
            max_score_idx = 0
            max_score = scores[0]
            for i in range(1, n_tops):
                if max_score < scores[i]:
                    max_score_idx = i
                    max_score = scores[i]
            best_pred_sent = ' '.join(top_preds[max_score_idx]['pred'])
            sf.write('{}\n'.format(best_pred_sent))
            sf.flush()







