# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import math
import random
import numpy as np
import json
from scipy.stats import pearsonr, spearmanr, kendalltau


def ndcg(y_true, y_pred, k, cnt=100, rel_threshold=0.):
    results = 0.
    for t in range(cnt):
        if k <= 0.:
            return 0.
        s = 0.
        c = zip(y_true, y_pred)
        random.shuffle(c)
        c_g = sorted(c, key=lambda x: x[0], reverse=True)
        c_p = sorted(c, key=lambda x: x[1], reverse=True)
        idcg = 0.
        ndcg = 0.
        for i, (g, p) in enumerate(c_g):
            if i >= k:
                break
            if g > rel_threshold:
                idcg += g / math.log(2. + i)
        for i, (g, p) in enumerate(c_p):
            if i >= k:
                break
            if g > rel_threshold:
                ndcg += g / math.log(2. + i)
        if idcg == 0.:
            results += 1.
        else:
            results += ndcg / idcg
    return results/float(cnt)


def eva_ranking_performance(rel, usefulness, pred_rel):
    rel_per, use_per = [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]
    rel_per[0] = ndcg(rel, pred_rel, 1)
    rel_per[1] = ndcg(rel, pred_rel, 3)
    rel_per[2] = ndcg(rel, pred_rel, 5)
    rel_per[3] = ndcg(rel, pred_rel, 10)
    rel_per[4] = ndcg(rel, pred_rel, len(rel))
    use_per[0] = ndcg(usefulness, pred_rel, 1)
    use_per[1] = ndcg(usefulness, pred_rel, 3)
    use_per[2] = ndcg(usefulness, pred_rel, 5)
    use_per[3] = ndcg(usefulness, pred_rel, 10)
    use_per[4] = ndcg(usefulness, pred_rel, len(usefulness))
    return rel_per, use_per


def eva_all(qid_list, did_list):
    rel_p, use_p = [], []
    for qid in qid_list:
        if qid in ['0']:
            continue
        rel, usefulness, pred = qid_list[qid]
        p1, p2 = eva_ranking_performance(rel, usefulness, pred)
        rel_p.append(p1)
        use_p.append(p2)
    print rel_p
    print use_p
    print 'query_p\tnDCG@1\tnDCG@3\tnDCG@5\tnDCG@10\tnDCG@All'
    print '\t'.join(['rel'] + map(str, np.array(rel_p, dtype=np.float32).mean(axis=0).tolist()))
    print '\t'.join(['use'] + map(str, np.array(use_p, dtype=np.float32).mean(axis=0).tolist()))


def evaluate_passage_ranking(input_file, score_index):
    # get labels
    with open('../data/doc_content_merged_anno.json', 'r') as f:
        data = json.load(f)
    pid_labels = {}
    for docid in data:
        pass_num = len(data[docid][0])
        for i in range(pass_num):
            pid = docid + '-' + str(i)
            rel = min(1, int(data[docid][4][i]))
            if rel == 0:
                use = 0
            else:
                use = int(data[docid][5][i])
            if pid not in pid_labels:
                pid_labels[pid] = [rel, use]
            else:
                print 'get label error'

    # get preds
    pid_scores = {}
    for line in open(input_file):
        arr = line.strip().split('\t')
        ids = json.loads(arr[2])
        scores = json.loads(arr[score_index])
        for i, pid in enumerate(ids):
            score = scores[i]
            if pid not in pid_scores:
                pid_scores[pid] = []
            pid_scores[pid].append(score)

    qid_list, did_list = {}, {}
    for pid in pid_scores:
        score = np.mean(pid_scores[pid])
        qid = pid.split('-')[0]
        did = pid.split('-')[0] + '-' + pid.split('-')[1]
        if qid not in qid_list:
            qid_list[qid] = [[], [], []]
        qid_list[qid][0].append(pid_labels[pid][0])
        qid_list[qid][1].append(pid_labels[pid][1])
        qid_list[qid][2].append(score)
        if did not in did_list:
            did_list[did] = [[], [], []]
        did_list[did][0].append(pid_labels[pid][0])
        did_list[did][1].append(pid_labels[pid][1])
        did_list[did][2].append(score)

    print len(qid_list), len(did_list)
    eva_all(qid_list, did_list)


def evaluate_docRel_estimation(input_file, score_index):
    # get preds
    did_scores = {}
    for line in open(input_file):
        arr = line.strip().split('\t')
        did = arr[0].split(';')[1]
        scores = json.loads(arr[-1])[-1]
        if did not in did_scores:
            did_scores[did] = []
        did_scores[did].append(scores)

    qid_list = {}
    label_list, pred_list = [], []
    for did in did_scores:
        tmp = np.mean(did_scores[did], axis=0)
        score = tmp[2]
        qid = did.split('-')[0]
        if qid not in qid_list:
            qid_list[qid] = [[], []]
        rel = int(did.split('-')[1])
        qid_list[qid][0].append(rel)
        qid_list[qid][1].append(score)
        label_list.append(rel)
        pred_list.append(score)


    print len(qid_list)
    print pearsonr(label_list, pred_list)
    print spearmanr(label_list, pred_list)


def main():
    evaluate_passage_ranking('../output/weibull/relevance_estimation.txt', 4)
    evaluate_docRel_estimation('../output/weibull/relevance_estimation.txt', -1)


if __name__ == '__main__':
    main()
