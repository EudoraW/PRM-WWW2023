# -*- coding: utf-8 -*-
from inference import ClickModel
import random
import sys
from collections import defaultdict
from datetime import datetime
from viewport_time_model import *
from .config_sample import *
from viewport_time_model_mer import ComplexNormalViewportTimeModelMRV3, ComplexWeibullViewportTimeModelMRV3, ComplexGammaViewportTimeModelMRV3


class MerVptModelV3(ClickModel):
    def __init__(self, ignoreIntents=True, ignoreLayout=True,
                 ignoreVerticalType=True, ignoreClickNecessity=True,
                 ignoreClickSatisfaction=True, ignoreExamSatisfaction=False,
                 useViewportTime=True, viewport_time_model=0,
                 config=None):

        self.ignoreVerticalType = ignoreVerticalType
        self.ignoreClickNecessity = ignoreClickNecessity
        self.ignoreClickSatisfaction = ignoreClickSatisfaction
        self.ignoreExamSatisfaction = ignoreExamSatisfaction
        self.useViewportTime = useViewportTime
        self.viewport_time_model = viewport_time_model

        ClickModel.__init__(self, ignoreIntents, ignoreLayout, config)
        print >> sys.stderr, 'McmVptModel:' + \
                             ' ignoreLayout=' + str(self.ignoreLayout) + \
                             ' ignoreVerticalType=' + str(self.ignoreVerticalType) + \
                             ' ignoreClickNecessity=' + str(self.ignoreClickNecessity) + \
                             ' ignoreClickSatisfaction=' + str(self.ignoreClickSatisfaction) + \
                             ' ignoreExamSatisfaction=' + str(self.ignoreExamSatisfaction) + \
                             ' useViewportTime=' + str(self.useViewportTime) + \
                             ' viewportTimeModel=' + str(self.viewport_time_model)

        # does not support IA model
        assert ignoreIntents
        assert useViewportTime
        if useViewportTime:
            if self.viewport_time_model == 8:
                self.viewport_time_model = ComplexNormalViewportTimeModelMRV3(
                    self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))
                print >> sys.stderr, 'VIEWPORT_TIME_MODEL:', 'ComplexNormalViewportTimeModelMRV3'
            if self.viewport_time_model == 10:
                self.viewport_time_model = ComplexWeibullViewportTimeModelMRV3(
                    self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))
                print >> sys.stderr, 'VIEWPORT_TIME_MODEL:', 'ComplexWeibullViewportTimeModelMRV3'
            if self.viewport_time_model == 9:
                self.viewport_time_model = ComplexGammaViewportTimeModelMRV3(
                    self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))
                print >> sys.stderr, 'VIEWPORT_TIME_MODEL:', 'ComplexGammaViewportTimeModelMRV3'


        # default UBM gamma
        self.gammaTypesNum = 2
        # UBM-layout, separate gamma for each vertical type
        # if self.ignoreClickNecessity and not self.ignoreVerticalType:
        #     self.gammaTypesNum = self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID)
        #     self.getGamma = self.getGammaWithVerticalId

    def train(self, sessions, test_sessions=None):
        # initialize alpha, beta, gamma, s_c, s_e #
        max_query_id = self.config.get('MAX_QUERY_ID')
        if max_query_id is None:
            print >> sys.stderr, 'WARNING: no MAX_QUERY_ID specified for', self
            max_query_id = 100000
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        # alpha: intent -> query -> url -> "attractiveness probability"
        self.alpha = dict((i,
                           [defaultdict(lambda: self.config.get('DEFAULT_ALPHA_V3', DEFAULT_ALPHA_V3)) \
                            for q in xrange(max_query_id)]) for i in possibleIntents)
        # gamma: freshness of the current result: gammaType -> rank -> "distance from the last click" - 1 -> examination probability
        self.gamma = [[[self.config.get('DEFAULT_GAMMA_V3', DEFAULT_GAMMA_V3) \
                        for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))] \
                       for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))],
                      [[1.0 \
                        for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))] \
                       for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]]
        # beta:
        self.beta = [self.config.get('DEFAULT_BETA_V3', DEFAULT_BETA_V3) for _ in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]

        self.theta = [[self.config.get('DEFAULT_THETA_0_V3', DEFAULT_THETA_0_V3) for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))], 
                      [self.config.get('DEFAULT_THETA_1_V3', DEFAULT_THETA_1_V3) for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))], 
                      [self.config.get('DEFAULT_THETA_2_V3', DEFAULT_THETA_2_V3) for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]]
        
        # s_e: prob. of satisfaction after examining a result that does not need click
        self.s_0 = self.config.get('DEFAULT_SAT_0_V3', DEFAULT_SAT_0_V3)
        self.s_mu = self.config.get('DEFAULT_SAT_MU_V3', DEFAULT_SAT_0_V3)
        self.s_1 = self.config.get('DEFAULT_SAT_1_V3', DEFAULT_SAT_0_V3)
        
        
        # start training #
        if not self.config.get('PRETTY_LOG', PRETTY_LOG):
            print >> sys.stderr, '-' * 80
            print >> sys.stderr, 'Start. Current time is', datetime.now()
        for iteration_count in xrange(self.config.get('MAX_ITERATIONS', MAX_ITERATIONS)):
            self.queryIntentsWeights = defaultdict(lambda: [])
            # not like in DBN! xxxFractions[0] is a numerator while xxxFraction[1] is a denominator
            alphaFractions = dict((i, [defaultdict(lambda: list(self.config.get('ALPHA_PRIOR_V3', ALPHA_PRIOR_V3)))
                                       for q in xrange(max_query_id)]) for i in possibleIntents)
            gammaFractions = [[[list(self.config.get('GAMMA_PRIOR_V3', GAMMA_PRIOR_V3))
                                for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
                               for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))],
                              [[list(self.config.get('GAMMA_PRIOR_V3', GAMMA_PRIOR_V3))
                                for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
                               for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]]
            betaFractions = [list(self.config.get('BETA_PRIOR_V3', BETA_PRIOR_V3))
                                 for _ in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
            
            thetaFractions = [[list(self.config.get('THETA_0_PRIOR_V3', THETA_0_PRIOR_V3)) for _ in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))],
                              [list(self.config.get('THETA_1_PRIOR_V3', THETA_1_PRIOR_V3)) for _ in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))],
                              [list(self.config.get('THETA_2_PRIOR_V3', THETA_2_PRIOR_V3)) for _ in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]]

            s_0Fractions = self.config.get('SAT_0_PRIOR_V3', SAT_0_PRIOR_V3)
            s_muFractions = self.config.get('SAT_MU_PRIOR_V3', SAT_MU_PRIOR_V3)
            s_1Fractions = self.config.get('SAT_1_PRIOR_V3', SAT_1_PRIOR_V3)
            
            self.viewport_time_model.update_init()

            # E-step
            for s in sessions:
                query = s.query
                if self.useViewportTime:
                    vpts = s.extraclicks['w_vpt']
                else:
                    vpts = [0] * len(s.clicks)
                layout = [0] * len(s.layout) if self.ignoreLayout else s.layout
                
                alphaList, betaList, gammaList, thetaList, s_0List, s_muList, s_1List = self.get_session_parameters(s, layout)
                
                f, b, z = self.get_forward_backward(s.clicks, alphaList, betaList, gammaList, thetaList, s_0List, s_muList, s_1List, viewport_time=vpts)

                prevClick = -1
                for rank, (ep, result) in enumerate(zip(s.clicks, s.results)):
                    vpt = vpts[rank]
                    # f_{i-1}
                    f0 = 1.0 if rank == 0 else f[rank - 1][0]
                    f1 = 0.0 if rank == 0 else f[rank - 1][1]
                    f2 = 0.0 if rank == 0 else f[rank - 1][2]

                    if ep == 0:
                        P1 = self.viewport_time_model.P(vpt, EP=0)
                    else:
                        P2 = self.viewport_time_model.P(vpt, EP=1, E=0)
                        P3 = self.viewport_time_model.P(vpt, EP=1, E=1, A=0)
                        P4 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=0)
                        P5_0 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=0)
                        P5_1 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=1, S_e=1)
                        P5_2 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=2, S_e=2)
                        P6_0 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=1)
                        P6_1 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=1, S_e=2)
                        P7 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=2)

                    # update gamma and viewport_time_model
                    if ep == 0:
                        # P(E_i=x|EP_1..M, V_1...M) = px
                        self.getGamma(gammaFractions, rank, prevClick, layout, False)[0] += 0.0
                        self.getGamma(gammaFractions, rank, prevClick, layout, False)[1] += 1.0
                        self.viewport_time_model.update(1.0, vpt, EP=0)
                    else:
                        # P(E_i=x, S_i-1=y|EP_1..M, V_1...M) = pxy
                        p00 = f0 * (1 - gammaList[rank]) * betaList[rank] * P2 * b[rank][0]
                        p01 = f1 * (1 - gammaList[rank]) * betaList[rank] * P2 * b[rank][1]
                        p02 = f2 * (1 - gammaList[rank]) * betaList[rank] * P2 * b[rank][2]
                        p0 = p00 + p01 + p02

                        # P(E_i=1, A_i=x, R_i=y, S_i=z, S_i-1=a|EP_1..M, V_1...M) = qxyza
                        q0000 = f0 * gammaList[rank] * (1 - alphaList[rank]) * P3 * b[rank][0]
                        q1000 = f0 * gammaList[rank] * alphaList[rank] * (1 - thetaList[0][rank]) * P4 * b[rank][0]
                        q1100 = f0 * gammaList[rank] * alphaList[rank] * thetaList[0][rank] * (1 - s_0List[rank]) * P5_0 * b[rank][0]
                        q1110 = f0 * gammaList[rank] * alphaList[rank] * thetaList[0][rank] * s_0List[rank] * (1 - s_muList[rank]) * P6_0 * b[rank][1]
                        q1120 = f0 * gammaList[rank] * alphaList[rank] * thetaList[0][rank] * s_0List[rank] * s_muList[rank] * P7 * b[rank][2]

                        q0011 = f1 * gammaList[rank] * (1 - alphaList[rank]) * P3 * b[rank][1]
                        q1011 = f1 * gammaList[rank] * alphaList[rank] * (1 - thetaList[1][rank]) * P4 * b[rank][1]
                        q1111 = f1 * gammaList[rank] * alphaList[rank] * thetaList[1][rank] * (1 - s_1List[rank]) * P5_1 * b[rank][1]
                        q1121 = f1 * gammaList[rank] * alphaList[rank] * thetaList[1][rank] * s_1List[rank] * P6_1 * b[rank][2]

                        q0022 = f2 * gammaList[rank] * (1 - alphaList[rank]) * P3 * b[rank][2]
                        q1022 = f2 * gammaList[rank] * alphaList[rank] * (1 - thetaList[2][rank]) * P4 * b[rank][2]
                        q1122 = f2 * gammaList[rank] * alphaList[rank] * thetaList[2][rank] * P5_2 * b[rank][2]

                        p1 = q0000 + q1000 + q1100 + q1110 + q1120 + q0011 + q1011 + q1111 + q1121 + q0022 + q1022 + q1122

                        self.getGamma(gammaFractions, rank, prevClick, layout, False)[0] += p1 / z[rank]
                        self.getGamma(gammaFractions, rank, prevClick, layout, False)[1] += (p1 + p0) / z[rank]

                        self.viewport_time_model.update(p0 / z[rank], vpt, EP=1, E=0)
                        self.viewport_time_model.update((q0000 + q0011 + q0022) / z[rank], vpt, EP=1, E=1, A=0)
                        self.viewport_time_model.update((q1000 + q1011 + q1022) / z[rank], vpt, EP=1, E=1, A=1, R=0)
                        self.viewport_time_model.update(q1100 / z[rank], vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=0)
                        self.viewport_time_model.update(q1111 / z[rank], vpt, EP=1, E=1, A=1, R=1, S_e_pre=1, S_e=1)
                        self.viewport_time_model.update(q1122 / z[rank], vpt, EP=1, E=1, A=1, R=1, S_e_pre=2, S_e=2)
                        self.viewport_time_model.update(q1110 / z[rank], vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=1)
                        self.viewport_time_model.update(q1121 / z[rank], vpt, EP=1, E=1, A=1, R=1, S_e_pre=1, S_e=2)
                        self.viewport_time_model.update(q1120 / z[rank], vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=2)

                    # update alpha
                    if ep == 0:
                        alphaFractions[False][query][s.results[rank]][0] += alphaList[rank]
                        alphaFractions[False][query][s.results[rank]][1] += 1.0
                    else:
                        # P(A_i=x|EP_1..M, V_1...M) = px
                        p0 = (p00 + p01 + p02) * (1 - alphaList[rank])
                        p0 += q0000 + q0011 + q0022

                        alphaFractions[False][query][s.results[rank]][0] += 1.0 - p0 / z[rank]
                        alphaFractions[False][query][s.results[rank]][1] += 1.0

                    # update beta
                    if ep == 0:
                        betaFractions[rank][0] += 0.0
                        betaFractions[rank][1] += 1.0
                    else:
                        p1 = p00 + p01 + p02

                        p1 /= z[rank]
                        betaFractions[rank][0] += p1
                        betaFractions[rank][1] += p1

                    if ep == 1:
                        ''' update theta '''
                        p0 = q1000
                        p1 = q1100 + q1110 + q1120
                        thetaFractions[0][rank][0] += p1 / z[rank]
                        thetaFractions[0][rank][1] += (p0 + p1) / z[rank]

                        p0 = q1011
                        p1 = q1111 + q1121
                        thetaFractions[1][rank][0] += p1 / z[rank]
                        thetaFractions[1][rank][1] += (p0 + p1) / z[rank]

                        p0 = q1022
                        p1 = q1122
                        thetaFractions[2][rank][0] += p1 / z[rank]
                        thetaFractions[2][rank][1] += (p0 + p1) / z[rank]

                        '''update s_0, s_mu, s_1'''
                        p0 = q1100
                        p1 = q1110
                        p2 = q1120
                        s_0Fractions[0] += (p1 + p2) / z[rank]
                        s_0Fractions[1] += (p0 + p1 + p2) /z[rank]

                        s_muFractions[0] += p2 / z[rank]
                        s_muFractions[1] += (p1 + p2) / z[rank]

                        p1 = q1111
                        p2 = q1121
                        s_1Fractions[0] += p2 / z[rank]
                        s_1Fractions[1] += (p1 + p2) / z[rank]

                    # update prevClick
                    # if ep == 1:
                    #     prevClick = rank

            if not self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('E')

            # M-step
            sum_square_displacement = 0.0
            # gamma
            for g in xrange(self.gammaTypesNum):
                for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
                    for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
                        gF = gammaFractions[g][r][d]
                        new_gamma = gF[0] / gF[1]
                        sum_square_displacement += (self.gamma[g][r][d] - new_gamma) ** 2
                        self.gamma[g][r][d] = new_gamma
            # alpha
            for i in possibleIntents:
                for q in xrange(max_query_id):
                    for url, aF in alphaFractions[i][q].iteritems():
                        new_alpha = aF[0] / aF[1]
                        sum_square_displacement += (self.alpha[i][q][url] - new_alpha) ** 2
                        self.alpha[i][q][url] = new_alpha
            # beta
            for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
                new_cn = betaFractions[r][0] / betaFractions[r][1]
                sum_square_displacement += (self.beta[r] - new_cn) ** 2
                self.beta[r] = new_cn

            # theta
            for i in range(3):
                for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
                    new_theta = thetaFractions[i][r][0] / thetaFractions[i][r][1]
                    sum_square_displacement += (self.theta[i][r] - new_theta) ** 2
                    self.theta[i][r] = new_theta

            # s_0
            new_s_0 = s_0Fractions[0] / s_0Fractions[1]
            sum_square_displacement += (self.s_0 - new_s_0) ** 2
            self.s_0 = new_s_0
            # # s_mu
            # new_s_mu = s_muFractions[0] / s_muFractions[1]
            # sum_square_displacement += (self.s_mu - new_s_mu) ** 2
            # self.s_mu = new_s_mu
            # # s_e_1
            # new_s_1 = s_1Fractions[0] / s_1Fractions[1]
            # sum_square_displacement += (self.s_1 - new_s_1) ** 2
            # self.s_1 = new_s_1

            # update viewport model
            self.viewport_time_model.update_finalize()

            if not self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('M\n')
            rmsd = math.sqrt(sum_square_displacement)

            ll, per = self.test(sessions, reportPositionPerplexity=False)
            if test_sessions is None:
                _ll, _per = 0., 0.
            else:
                _ll, _per = self.test(test_sessions, reportPositionPerplexity=False)
            if self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('Iteration: %d\t%f\t%f\t%f\t%f\n' % (iteration_count + 1, ll, per, _ll, _per))
            else:
                print >> sys.stderr, 'Iteration: %d, ERROR: %f %f %f %f %f' % (iteration_count + 1, rmsd, ll, per, _ll, _per)
        if self.config.get('PRETTY_LOG', PRETTY_LOG):
            sys.stderr.write('\n')

    def get_session_parameters(self, session, layout):
        """return alphaList, betaList, gammaList, s_cList, s_0List"""
        alphaList = []
        betaList = []
        gammaList = []
        thetaList = [[], [], []]
        s_0List = []
        s_muList = []
        s_1List = []

        query = session.query
        prevClick = -1
        for rank, url in enumerate(session.results):
            alphaList.append(self.alpha[False][query][url])

            gammaList.append(self.getGamma(self.gamma, rank, prevClick, layout, False))

            betaList.append(self.beta[rank])
            
            thetaList[0].append(self.theta[0][rank])
            thetaList[1].append(self.theta[1][rank])
            thetaList[2].append(self.theta[2][rank])

            s_0List.append(self.s_0)
            s_muList.append(self.s_mu)
            s_1List.append(self.s_1)

        return alphaList, betaList, gammaList, thetaList, s_0List, s_muList, s_1List

    def get_relevance_parameters(self, query, url, rank):
        """return alpha, beta, s_c, s_e"""
        return self.alpha[False][query][url], \
               self.beta[rank], \
               self.gamma[0][rank][rank], \
               self.theta[0][rank], \
               self.theta[1][rank], \
               self.theta[2][rank], \
               self.s_0, \
               self.s_mu, \
               self.s_1

    def get_forward_backward(self, clicks, alphaList, betaList, gammaList, thetaList, s_0List, s_muList, s_1List, **kwargs):
        """return f, b, z"""
        M = len(clicks)
        vpts = kwargs['viewport_time']
        
        f = [[1.0, 0., 0.]]
        b = [[1.0, 1.0, 1.0] for i in xrange(M)]
        z = [1.0]

        # forward
        eps = clicks
        for i, ep in enumerate(eps):
            vpt = vpts[i]
            if ep == 0:
                P1 = self.viewport_time_model.P(vpt, EP=0)
                t0 = f[i][0] * (1.0 - gammaList[i]) * (1.0 - betaList[i]) * P1
                t1 = f[i][1] * (1.0 - gammaList[i]) * (1.0 - betaList[i]) * P1
                t2 = f[i][2] * (1.0 - gammaList[i]) * (1.0 - betaList[i]) * P1
                f.append([t0, t1,  t2])
            else:
                P2 = self.viewport_time_model.P(vpt, EP=1, E=0)
                P3 = self.viewport_time_model.P(vpt, EP=1, E=1, A=0)
                P4 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=0)
                P5_0 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=0)
                P5_1 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=1, S_e=1)
                P5_2 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=2, S_e=2)
                P6_0 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=1)
                P6_1 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=1, S_e=2)
                P7 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=2)

                t0 = (1.0 - gammaList[i]) * betaList[i] * P2
                t0 += gammaList[i] * ((1 - alphaList[i]) * P3
                                      + alphaList[i] * ((1 - thetaList[0][i]) * P4 + thetaList[0][i] * (1 - s_0List[i]) * P5_0))
                t0 *= f[i][0]

                t1 = f[i][0] * gammaList[i] * alphaList[i] * thetaList[0][i] * s_0List[i] * (1 - s_muList[i]) * P6_0
                t11 = (1 - alphaList[i]) * P3
                t11 += alphaList[i] * ((1 - thetaList[1][i]) * P4 + thetaList[1][i] * (1 - s_1List[i]) * P5_1)
                t11 *= gammaList[i]
                t11 += (1.0 - gammaList[i]) * betaList[i] * P2
                t11 *= f[i][1]
                t1 += t11

                t2 = f[i][0] * gammaList[i] * alphaList[i] * thetaList[0][i] * s_0List[i] * s_muList[i] * P7
                t2 += f[i][1] * gammaList[i] * alphaList[i] * thetaList[1][i] * s_1List[i] * P6_1
                t22 = (1 - alphaList[i]) * P3
                t22 += alphaList[i] * ((1 - thetaList[2][i]) * P4 + thetaList[2][i] * P5_2)
                t22 *= gammaList[i]
                t22 += (1.0 - gammaList[i]) * betaList[i] * P2
                t22 *= f[i][2]
                t2 += t22

                f.append([t0, t1, t2])

            z.append(sum(f[i + 1]))

            f[i + 1][0] /= z[i + 1]
            f[i + 1][1] /= z[i + 1]
            f[i + 1][2] /= z[i + 1]

        f = f[1:]
        z = z[1:]

        # backward
        # re-used p[i][t] computed in the forward pass
        for i in range(M - 2, -1, -1):
            vpt = vpts[i+1]
            if eps[i + 1] == 0:
                P1 = self.viewport_time_model.P(vpt, EP=0)
                b[i][0] = b[i + 1][0] * (1 - gammaList[i+1]) * (1 - betaList[i+1]) * P1
                b[i][1] = b[i + 1][1] * (1 - gammaList[i+1]) * (1 - betaList[i+1]) * P1
                b[i][2] = b[i + 1][2] * (1 - gammaList[i+1]) * (1 - betaList[i+1]) * P1
            else:
                P2 = self.viewport_time_model.P(vpt, EP=1, E=0)
                P3 = self.viewport_time_model.P(vpt, EP=1, E=1, A=0)
                P4 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=0)
                P5_0 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=0)
                P5_1 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=1, S_e=1)
                P5_2 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=2, S_e=2)
                P6_0 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=1)
                P6_1 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=1, S_e=2)
                P7 = self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=2)

                b00 = (1 - alphaList[i+1]) * P3
                b00 += alphaList[i+1] * ((1 - thetaList[0][i+1]) * P4 + thetaList[0][i+1] * (1 - s_0List[i+1]) * P5_0)
                b00 *= gammaList[i+1]
                b00 += (1 - gammaList[i+1]) * betaList[i+1] * P2
                b00 *= b[i + 1][0]

                b01 = b[i + 1][1] * gammaList[i+1] * alphaList[i+1] * thetaList[0][i+1] * s_0List[i+1] * (1 - s_muList[i+1]) * P6_0
                b02 = b[i + 1][2] * gammaList[i+1] * alphaList[i+1] * thetaList[0][i+1] * s_0List[i+1] * s_muList[i+1] * P7
                b[i][0] = b00 + b01 + b02

                b11 = (1 - alphaList[i+1]) * P3
                b11 += alphaList[i+1] * ((1 - thetaList[1][i+1]) * P4 + thetaList[1][i+1] * (1 - s_1List[i+1]) * P5_1)
                b11 *= gammaList[i+1]
                b11 += (1 - gammaList[i+1]) * betaList[i+1] * P2
                b11 *= b[i + 1][1]

                b12 = b[i + 1][2] * gammaList[i+1] * alphaList[i+1] * thetaList[1][i+1] * s_1List[i+1] * P6_1
                b[i][1] = b11 + b12

                b22 = (1 - alphaList[i+1]) * P3
                b22 += alphaList[i+1] * ((1 - thetaList[2][i+1]) * P4 + thetaList[2][i+1] * P5_2)
                b22 *= gammaList[i+1]
                b22 += (1 - gammaList[i+1]) * betaList[i+1] * P2
                b22 *= b[i + 1][2]
                b[i][2] = b22

            b[i] = [b[i][0] / z[i + 1], b[i][1] / z[i + 1], b[i][2] / z[i + 1]]

        return f, b, z

    def test_forward_backward(self, clicks=[1, 0, 0, 1, 0], viewport_time=[1000, 1000, 1000, 1000, 0]):
        M = len(clicks)
        alphaList = [0.5] * M
        print 'alpha: ' + ' '.join(["%.3f" % x for x in alphaList])
        betaList = [0.5] * M
        print 'beta: ' + ' '.join(["%.3f" % x for x in betaList])
        gammaList = list(reversed([0.1 * i for i in range(1, M + 1)]))
        print 'gamma: ' + ' '.join(["%.3f" % x for x in gammaList])
        s_cList = [0.8] * M
        print 's_cList: ' + ' '.join(["%.3f" % x for x in s_cList])
        s_eList = [0.5] * M
        print 's_eList: ' + ' '.join(["%.3f" % x for x in s_eList])
        f, b, z = self.get_forward_backward(clicks, alphaList, betaList, gammaList, s_cList, s_eList,
                                            viewport_time=viewport_time)
        print 'results:'
        print 'f0:\t' + ' '.join(["%.3f" % x[0] for x in f])
        print 'f1:\t' + ' '.join(["%.3f" % x[1] for x in f])
        print 'z:\t' + ' '.join(["%.3f" % x for x in z])

        print 'b0:\t' + ' '.join(["%.3f" % x[0] for x in b])
        print 'b1:\t' + ' '.join(["%.3f" % x[1] for x in b])

    def _getSessionProb(self, s):
        clickProbs = self._get_click_probs(s, [False, True])
        N = len(s.clicks)
        return clickProbs[False][N - 1] / clickProbs[True][N - 1]

    @staticmethod
    def getGamma(gammas, k, prevClick, layout, intent):
        index = 1 if layout[k] else 0
        return gammas[index][k][k - prevClick - 1]

    def _get_click_probs(self, s, possibleIntents):
        """
            Returns exposureProbs list
            exposureProbs[i][k] = P(V_1, ..., V_k | I=i)
        """

        vpts = [0] * len(s.clicks)

        if self.useViewportTime:
            vpts = s.extraclicks['w_vpt']

        layout = [0] * len(s.layout) if self.ignoreLayout else s.layout
        alphaList, betaList, gammaList, thetaList, s_0List, s_muList, s_1List = self.get_session_parameters(s, layout)
        f, b, z = self.get_forward_backward(s.clicks, alphaList, betaList, gammaList, thetaList, s_0List, s_muList, s_1List, viewport_time=vpts)


        exposureProbs = [1.0]
        for i, ep in enumerate(s.clicks):
            f0, f1, f2 = (f[i - 1][0], f[i - 1][1], f[i - 1][2]) if i >= 1 else (1.0, 0.0, 0.0)
            vpt = vpts[i]

            P1 = 1. # self.viewport_time_model.P(vpt, EP=0)
            P2 = 1. # self.viewport_time_model.P(vpt, EP=1, E=0)
            P3 = 1. # self.viewport_time_model.P(vpt, EP=1, E=1, A=0)
            P4 = 1. # self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=0)
            P5_0 = 1. # self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=0)
            P5_1 = 1. # self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=1, S_e=1)
            P5_2 = 1. # self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=2, S_e=2)
            P6_0 = 1. # self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=1)
            P6_1 = 1. # self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=1, S_e=2)
            P7 = 1. # self.viewport_time_model.P(vpt, EP=1, E=1, A=1, R=1, S_e_pre=0, S_e=2)

            # P(EP_i=0, S_i-1=x) = px
            p0 = f0 * (1 - gammaList[i]) * (1 - betaList[i]) * P1
            p1 = f1 * (1 - gammaList[i]) * (1 - betaList[i]) * P1
            p2 = f2 * (1 - gammaList[i]) * (1 - betaList[i]) * P1

            # P(E_i=x, S_i-1=y|EP_1..M, V_1...M) = pxy
            p00 = f0 * (1 - gammaList[i]) * betaList[i] * P2
            p01 = f1 * (1 - gammaList[i]) * betaList[i] * P2
            p02 = f2 * (1 - gammaList[i]) * betaList[i] * P2

            # P(E_i=1, A_i=x, R_i=y, S_i=z, S_i-1=a|EP_1..M, V_1...M) = qxyza
            q0000 = f0 * gammaList[i] * (1 - alphaList[i]) * P3
            q1000 = f0 * gammaList[i] * alphaList[i] * (1 - thetaList[0][i]) * P4
            q1100 = f0 * gammaList[i] * alphaList[i] * thetaList[0][i] * (1 - s_0List[i]) * P5_0
            q1110 = f0 * gammaList[i] * alphaList[i] * thetaList[0][i] * s_0List[i] * (1 - s_muList[i]) * P6_0
            q1120 = f0 * gammaList[i] * alphaList[i] * thetaList[0][i] * s_0List[i] * s_muList[i] * P7

            q0011 = f1 * gammaList[i] * (1 - alphaList[i]) * P3
            q1011 = f1 * gammaList[i] * alphaList[i] * (1 - thetaList[1][i]) * P4
            q1111 = f1 * gammaList[i] * alphaList[i] * thetaList[1][i] * (1 - s_1List[i]) * P5_1
            q1121 = f1 * gammaList[i] * alphaList[i] * thetaList[1][i] * s_1List[i] * P6_1

            q0022 = f2 * gammaList[i] * (1 - alphaList[i]) * P3
            q1022 = f2 * gammaList[i] * alphaList[i] * (1 - thetaList[2][i]) * P4
            q1122 = f2 * gammaList[i] * alphaList[i] * thetaList[2][i] * P5_2


            total = p0 + p1 + p2 + \
                    p00 + p01 + p02 + \
                    q0000 + q1000 + q1100 + q1110 + q1120 + q0011 + q1011 + q1111 + q1121 + q0022 + q1022 + q1122

            if ep == 0:
                p = (p0 + p1 + p2) / total
            else:
                p = 1. - (p0 + p1 + p2) / total
            exposureProbs.append(exposureProbs[-1] * p)

        return dict((i, exposureProbs[1:]) for i in possibleIntents)




