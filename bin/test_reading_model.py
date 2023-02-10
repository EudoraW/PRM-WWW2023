#!/usr/bin/env pypy

import sys
import glob
import argparse
import numpy as np
import os

sys.path.insert(0, "..")
from clickmodels.inference import *
from clickmodels.input_reader import InputReader
from collections import Counter, defaultdict
import math
import json

from clickmodels.config_sample import *
from clickmodels.MerVptModelV3 import MerVptModelV3


def DCG(rels, at=5):
    rels = [1.0 * r if r >= 0.0 else 0.0 for r in rels][0:at]
    rels = [2**r - 1.0 for r in rels]
    discount = [math.log(i+2, 2) for i in xrange(at)]
    ret = [r / d for r, d in zip(rels, discount)]
    for i in xrange(1, min(at, len(ret))):
        ret[i] += ret[i - 1]
    return ret

def read_df(filename, min_f=1, max_f=10000000):
    data = []
    for line in open(filename):
        e = line.split('\t')
        f = int(e[1])
        if f < min_f:
            continue
        if f > max_f:
            continue
        data.append([
                0,
                int(e[1]),
                float(e[2]),
                float(e[3])
        ]) # + json.loads(e[4]) + json.loads(e[5]))
    ret = np.array(data, dtype=np.float64)
    print >> sys.stderr, 'Test set size:', len(ret)
    return ret

FOLD_NUM = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', help="path to the training set")
    parser.add_argument('test_dir', help="path to the test set")
    parser.add_argument('-o', '--output', help="path to output directory")
    # parser.add_argument('-r', '--relevance_file',
    #                     help="if relevance file is given, the ndcg file for each test session will be computed")
    parser.add_argument('-m', '--model',
                        help='the name of click model [default=MCM]',
                        default='MCM')
    parser.add_argument('-N', '--num_train_files',
                        help='the first N training files will be used, 0 means use all the files [default=1]',
                        type=int,
                        default=1)
    parser.add_argument('-M', '--num_test_files',
                        help='the first M test files will be used, 0 means use all the files [default=1]',
                        type=int,
                        default=1)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='increase output verbosity.')
    parser.add_argument('-i', '--ignore_no_clicks', action='store_true',
                        help='ignore sessions that have no clicks')
    parser.add_argument('-c', '--configs', help='additional configs')
    parser.add_argument('-t', '--viewport_time', action='store_true',
                        help='use viewport time data to train the model')
    parser.add_argument('-I', '--ignore_no_viewport', action='store_true',
                        help='ignore sessions with zero viewport time')
    parser.add_argument('-V', '--viewport_time_model', default=0,
                        type=int, help='choose viewport time model')
    parser.add_argument('-f', '--query_frequency', default=1,
                        type=int, help='the query frequency threshold for test')

    args = parser.parse_args()

    MODEL_CONSTRUCTORS = {
        'DBN': lambda config: DbnModel((0.9, 0.9, 0.9, 0.9), config=config),
        # 'DBN-layout': lambda config: DbnModel((1.0, 0.9, 1.0, 0.9), ignoreLayout=False, config=config),
        'UBM': lambda config: UbmModel(config=config),
        'DCM': lambda config: DcmModel(config=config),
        'UBM-layout': lambda config: UbmModel(ignoreLayout=False, ignoreVerticalType=False, config=config),
        'EBUBM': lambda config: UbmModel(explorationBias=True, config=config),
        'UBM-N': lambda config: McmModel(ignoreClickSatisfaction=True, ignoreExamSatisfaction=True, config=config),
        'UBM-CS': lambda config: McmModel(ignoreClickNecessity=True, ignoreExamSatisfaction=True, config=config),
        'MCM': lambda config: McmModel(config=config),
        'MCM-VPT': lambda config: McmVptModel(config=config, viewport_time_model=args.viewport_time_model),
        'MCM-VPT-OFF': lambda config: McmVptModel(config=config, useViewportTime=False),
        'MER-VPT-V3': lambda config: MerVptModelV3(config=config, viewport_time_model=args.viewport_time_model),
    }

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    else:
        print 'delete', args.output
        os.rmdir(args.output)
        os.mkdir(args.output)

    for cross in range(FOLD_NUM):

        readInput = InputReader(MIN_DOCS_PER_QUERY, MAX_DOCS_PER_QUERY,
                                EXTENDED_LOG_FORMAT, SERP_SIZE,
                                TRAIN_FOR_METRIC,
                                discard_no_clicks=args.ignore_no_clicks,
                                read_viewport_time=args.viewport_time,
                                discard_no_viewport=args.ignore_no_viewport)

        print 'cross ' + str(cross) + ': prepare sessions data...'

        train_files = [args.train_dir + '/all_data.tsv']
        test_files = [args.train_dir + '/all_data.tsv']

        train_sessions = []
        for fileNumber in xrange(len(train_files)):
            f = train_files[fileNumber]
            new_sessions = readInput(open(f))
            readInput.get_vertical_id_mapping(new_sessions)
            train_sessions += new_sessions
        print >> sys.stderr, 'Train set size:', len(train_sessions)
        max_train_query_id = readInput.current_query_id
        query_freq = Counter(s.query for s in train_sessions)

        test_sessions = []
        for fileNumber in xrange(len(test_files)):
            f = test_files[fileNumber]
            new_sessions = readInput(open(f))
            readInput.get_vertical_id_mapping(new_sessions)
            test_sessions += new_sessions

        print 'train click model...'
        config = {
            'MAX_QUERY_ID': readInput.current_query_id + 1,
            'MAX_ITERATIONS': MAX_ITERATIONS,
            'DEBUG': DEBUG,
            'PRETTY_LOG': not args.verbose,
            'MAX_DOCS_PER_QUERY': MAX_DOCS_PER_QUERY,
            'SERP_SIZE': SERP_SIZE,
            'TRANSFORM_LOG': TRANSFORM_LOG,
            'QUERY_INDEPENDENT_PAGER': QUERY_INDEPENDENT_PAGER,
            'DEFAULT_REL': DEFAULT_REL,
            'MAX_VERTICAL_ID': readInput.max_vertical_id,
        }

        if args.configs:
            import json
            additional_configs = json.loads(args.configs)
            config.update(additional_configs)
            print config

        model_cls = MODEL_CONSTRUCTORS[args.model]
        m = model_cls(config)
        m.train(train_sessions)

        if args.output is not None:

            pred_fout = open(args.output + '/click_prediction.txt', 'a')

            for fileNumber in xrange(len(test_files)):
                f = test_files[fileNumber]
                for line in open(f):
                    # get sess and line
                    sess = readInput([line])
                    if len(sess) == 0:
                        continue
                    sess = sess[0]
                    # skip the test session if the query was not seen in training set
                    if sess.query >= max_train_query_id:
                        continue

                    # click prediction
                    entries = line.rstrip().split('\t')
                    sid = entries[0]
                    _ll, _position_ll = m.test_one_by_one([sess])
                    print >>pred_fout, '%s\t%d\t%f\t%s\t%s' % (
                        sid,
                        query_freq[sess.query],
                        _ll,
                        _position_ll,
                        str(sess.clicks)
                    )

            pred_fout.close()

        # data = []
        # df = read_df(args.output + '/click_prediction.txt', min_f=args.query_frequency)
        # data = df
        # data = np.transpose(data)
        # print np.mean(data[1]), np.mean(data[2]), np.mean(data[3])
        data = []
        for line in open(args.output + '/click_prediction.txt'):
            e = line.split('\t')
            data.append([
                0,
                int(e[1]),
                float(e[2]),
                json.loads(e[3])
            ])

        data = data[cross*1620/FOLD_NUM: (cross+1)*1620/FOLD_NUM]
        print >> sys.stderr, 'Test set size:', len(data)
        data = np.transpose(data)
        print np.mean(data[2]),
        position_ppl = [[] for _ in range(config['MAX_DOCS_PER_QUERY'])]
        for i in range(len(data[3])):
            for j in range(len(data[3][i])):
                position_ppl[j].append(data[3][i][j] if j == 0 else (data[3][i][j] - data[3][i][j-1]))
        for i in range(len(position_ppl)):
            position_ppl[i] = 2 ** (0. - np.mean(position_ppl[i]))
        print np.mean(position_ppl), position_ppl


        if args.output is not None:
            rel_fout = open(args.output + '/relevance_estimation.txt', 'a')

            for fileNumber in xrange(len(train_files)):
                f = train_files[fileNumber]
                for line in open(f):
                    # get sess and line
                    sess = readInput([line])
                    if len(sess) == 0:
                        continue
                    sess = sess[0]

                    entries = line.rstrip().split('\t')
                    sid = entries[0]

                    if args.model == 'MER-VPT-V2':
                        alphaList, betaList, gammaList, s_cList, s_eList, s_e_1List, s_e_2List = m.get_session_parameters(sess, [0 for _ in range(len(sess.clicks))])
                        f, b, z = m.get_forward_backward(sess.clicks, alphaList, betaList, gammaList, s_cList, s_eList, s_e_1List, s_e_2List,
                                                         viewport_time=sess.extraclicks['w_vpt'])

                        alpha, beta, gamma, gamma1, s_e, s_e_1, s_e_2 = [], [], [], [], [], [], []
                        i = 0
                        for url, vrid in zip(sess.results, sess.layout):
                            _a, _b, _c, _c1, _s_e, _s_e_1, _s_e_2 = m.get_relevance_parameters(sess.query, url, i)
                            alpha.append(_a)
                            beta.append(_b)
                            gamma.append(_c)
                            gamma1.append(_c1)
                            s_e.append(_s_e)
                            s_e_1.append(_s_e_1)
                            s_e_2.append(_s_e_2)
                            i += 1
                        query = entries[1]
                        urls = json.loads(entries[4])
                        print >>rel_fout, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
                            sid, query,
                            entries[4], # results
                            entries[6], # exposure
                            str(alpha),
                            str(beta),
                            str(gamma),
                            str(gamma1),
                            str(s_e),
                            str(s_e_1),
                            str(s_e_2),
                            str(f)
                        )
                    elif args.model == 'MER-VPT':
                        alpha, beta, gamma, s_c, s_e = m.get_session_parameters(sess, [0 for _ in range(len(sess.clicks))])
                        f, b, z = m.get_forward_backward(sess.clicks, alpha, beta, gamma, s_c, s_e, viewport_time=sess.extraclicks['w_vpt'])
                        alpha, beta, gamma, s_e = [], [], [], []
                        i = 0
                        for url, vrid in zip(sess.results, sess.layout):
                            _a, _b, _c, _s_e = m.get_relevance_parameters(sess.query, url, i)
                            alpha.append(_a)
                            beta.append(_b)
                            gamma.append(_c)
                            s_e.append(_s_e)
                            i += 1
                        query = entries[1]
                        urls = json.loads(entries[4])
                        print >> rel_fout, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
                            sid, query,
                            entries[4],  # results
                            entries[6],  # exposure
                            str(alpha),
                            str(beta),
                            str(gamma),
                            str(s_e),
                            str(f)
                        )
                    elif args.model == 'MER-VPT-V3':

                        alphaList, betaList, gammaList, thetaList, s_0List, s_muList, s_1List = m.get_session_parameters(sess, [0 for _ in range(len(sess.clicks))])
                        f, b, z = m.get_forward_backward(sess.clicks, alphaList, betaList, gammaList, thetaList, s_0List, s_muList, s_1List, viewport_time=sess.extraclicks['w_vpt'])

                        # i = 0
                        # for url, vrid in zip(sess.results, sess.layout):
                        #     _a, _b, _c, _t_0, _t_1, _t_2, _s_e, _s_e_1, _s_e_2 = m.get_relevance_parameters(sess.query, url, i)
                        #     alpha.append(_a)
                        #     beta.append(_b)
                        #     gamma.append(_c)
                        #     theta0.append(_t_0)
                        #     theta1.append(_t_1)
                        #     theta2.append(_t_2)
                        #     s_e.append(_s_e)
                        #     s_e_1.append(_s_e_1)
                        #     s_e_2.append(_s_e_2)
                        #     i += 1
                        query = entries[1]
                        urls = json.loads(entries[4])
                        print >> rel_fout, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
                            sid, query,
                            entries[4],  # results
                            entries[6],  # exposure
                            str(alphaList),
                            str(betaList),
                            str(gammaList),
                            str(thetaList[0]),
                            str(thetaList[1]),
                            str(thetaList[2]),
                            str(s_0List),
                            str(s_muList),
                            str(s_1List),
                            str(f)
                        )

            rel_fout.close()
