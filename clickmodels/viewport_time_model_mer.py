# -*- coding: utf-8 -*-
import numpy as np
import math

from .config_sample import *
from viewport_time_model import DefaultViewportTimeModel


class ComplexNormalViewportTimeModelMRV3(DefaultViewportTimeModel):
    def __init__(self, max_vertical_id):
        super(ComplexNormalViewportTimeModelMRV3, self).__init__()
        self.scale = 1000. # 可以自己调调
        self.epsilon = 0.1 / self.scale
        self.time_threshold = self.convert_v(VPT_EPSILON)
        self.max_vertical_id = 1 # max_vertical_id

        self.s_ep0 = [0.75 for _ in xrange(self.max_vertical_id)] # sigma
        self.m_ep0 = [1. for _ in xrange(self.max_vertical_id)] # mu
        self.s_ep1_e0 = [1.09 for _ in xrange(self.max_vertical_id)]
        self.m_ep1_e0 = [0.31 for _ in xrange(self.max_vertical_id)]
        self.s_ep1_e1_r0 = [0.57 for _ in xrange(self.max_vertical_id)]
        self.m_ep1_e1_r0 = [1.54 for _ in xrange(self.max_vertical_id)]
        self.s_ep1_e1_r1_s0 = [0.57 for _ in xrange(self.max_vertical_id)]
        self.m_ep1_e1_r1_s0 = [2.45 for _ in xrange(self.max_vertical_id)]
        self.s_ep1_e1_r1_s1 = [1.2 for _ in xrange(self.max_vertical_id)]
        self.m_ep1_e1_r1_s1 = [-1. for _ in xrange(self.max_vertical_id)]
        self.s_ep1_e1_r1_s2 = [1.2 for _ in xrange(self.max_vertical_id)]
        self.m_ep1_e1_r1_s2 = [-1. for _ in xrange(self.max_vertical_id)]

        self.p_ep0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s1 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s1 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s2 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s2 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_origin = 0

    def get_method_name(self):
        return 'ComplexNormalViewportTimeModelMRV3'

    def normal_quality_function(self, x, s, m):
        # return math.exp(-(x - m) ** 2 / (2 * s ** 2)) / (math.sqrt(2*math.pi) * s)
        return math.exp(-(math.log(x) - m) ** 2 / (2 * s ** 2)) / (math.sqrt(2 * math.pi) * s * x)

    def convert_v(self, v):
        # v = math.log(max(self.epsilon, v/self.scale))
        v = max(self.epsilon, v / self.scale)
        return v

    def P(self, v, **kwargs):
        # return 1.0
        self.v_origin = v
        v = self.convert_v(v)
        layout = 0 # layout = kwargs['layout']
        if 'EP' in kwargs and kwargs['EP'] == 0:
            # if 'C' in kwargs and kwargs['C'] == 1:
            #     return self.epsilon
            # elif v < self.time_threshold:
            #     return 100.
            # else:
            #     s = self.s_e0[layout]
            #     m = self.m_e0[layout]
            #     return max(self.normal_quality_function(v, s, m), self.epsilon)
            if v < self.time_threshold:
                return 100
            else:
                return 1e-12
        elif 'EP' in kwargs and kwargs['EP'] == 1:
            # v = math.log(max(20./self.scale, self.v_origin/self.scale))
            if 'E' in kwargs and kwargs['E'] == 0:
                s = self.s_ep1_e0[layout]
                m = self.m_ep1_e0[layout]
                # return max(self.normal_quality_function(v, s, m), self.epsilon)
            elif 'E' in kwargs and kwargs['E'] == 1:
                if 'R' in kwargs and kwargs['R'] == 1:
                    s = self.s_ep1_e1_r1_s0[layout]
                    m = self.m_ep1_e1_r1_s0[layout]
                    # return max(self.normal_quality_function(v, s, m), self.epsilon)
                else:
                    s = self.s_ep1_e1_r0[layout]
                    m = self.m_ep1_e1_r0[layout]
                    # return max(self.normal_quality_function(v, s, m), self.epsilon)
            else:
                raise ValueError('ComplexNormalViewportTimeModelMR P() input error!')
            return max(self.normal_quality_function(v, s, m), self.epsilon)
        else:
            raise ValueError('ComplexNormalViewportTimeModelMR P() input error!')

    def update_init(self):
        """
        initialize the training parameters before M-step
        Returns: None

        """
        self.p_ep0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s1 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s1 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s2 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s2 = [[] for _ in xrange(self.max_vertical_id)]

    def update(self, posterior, v, **kwargs):
        """
        update the training parameters when there is an item in the Q-function of the form
        P(conditions|C_1..M, V_1..M) * log P(V_i|conditions)
        Args:
            posterior:  posterior prob P(conditions|C_1..M, V_1..M) estimated in E-step
            v: viewport time
            **kwargs: conditions su

        Returns: None

        """
        v = self.convert_v(v)
        if v < self.time_threshold:
            return
        layout = 0 # kwargs['layout']
        posterior = max(self.epsilon, posterior)
        if 'EP' in kwargs and kwargs['EP'] == 1:
            if 'E' in kwargs and kwargs['E'] == 0:
                self.p_ep1_e0[layout].append(posterior)
                self.v_ep1_e0[layout].append(v)
            elif 'E' in kwargs and kwargs['E'] == 1:
                if 'R' in kwargs and kwargs['R'] == 1:
                    self.p_ep1_e1_r1_s0[layout].append(posterior)
                    self.v_ep1_e1_r1_s0[layout].append(v)
                else:
                    self.p_ep1_e1_r0[layout].append(posterior)
                    self.v_ep1_e1_r0[layout].append(v)

    def updata_s_m(self, e_p, e_v, s_orgin, m_orgin):
        # get average k and t
        min_num = 50
        s_list, m_list = [], []
        avg_s, avg_m = [], []
        for i in xrange(len(e_p)):
            # print len(e_p[i])
            if len(e_p[i]) >= min_num:
                avg_s.append(s_orgin[i])
                avg_m.append(m_orgin[i])
            else:
                raise ValueError('ComplexNormalViewportTimeModelMR updata_s_m() min num < 50')
        avg_s = np.mean(avg_s)
        avg_m = np.mean(avg_m)
        # update s, m
        for i in xrange(len(e_p)):
            if len(e_p[i]) < min_num:
                s_list.append(avg_s)
                m_list.append(avg_m)
                continue
            p = np.array(e_p[i], dtype=np.float64)
            v = np.array(e_v[i], dtype=np.float64)
            # m = np.sum(p*v) / np.sum(p)
            # s = math.sqrt(np.sum(p*np.power(v-m, 2))/np.sum(p))
            m = np.sum(p*np.log(v)) / np.sum(p)
            s = math.sqrt(np.sum(p*np.power(np.log(v)-m, 2))/np.sum(p))
            if math.fabs(s) < self.epsilon:
                s_list.append(self.epsilon)
                m_list.append(m)
                print '~~~'
                continue
            s_list.append(s)
            m_list.append(m)
        # print s_list
        # print m_list
        # print ''
        return s_list, m_list

    def update_finalize(self):
        """
        update the model parameters using the training parameters
        Returns: None

        """
        # self.s_ep0, self.m_ep0 = self.updata_s_m(self.p_ep0, self.v_ep0, self.s_ep0, self.m_ep0)
        self.s_ep1_e0, self.m_ep1_e0 = self.updata_s_m(self.p_ep1_e0, self.v_ep1_e0, self.s_ep1_e0, self.m_ep1_e0)
        self.s_ep1_e1_r0, self.m_ep1_e1_r0 = self.updata_s_m(self.p_ep1_e1_r0, self.v_ep1_e1_r0, self.s_ep1_e1_r0, self.m_ep1_e1_r0)
        self.s_ep1_e1_r1_s0, self.m_ep1_e1_r1_s0 = self.updata_s_m(self.p_ep1_e1_r1_s0, self.v_ep1_e1_r1_s0, self.s_ep1_e1_r1_s0, self.m_ep1_e1_r1_s0)
        # self.s_ep1_e1_r1_s1, self.m_ep1_e1_r1_s1 = self.updata_s_m(self.p_ep1_e1_r1_s1, self.v_ep1_e1_r1_s1, self.s_ep1_e1_r1_s1, self.m_ep1_e1_r1_s1)
        # self.s_ep1_e1_r1_s2, self.m_ep1_e1_r1_s2 = self.updata_s_m(self.p_ep1_e1_r1_s2, self.v_ep1_e1_r1_s2, self.s_ep1_e1_r1_s2, self.m_ep1_e1_r1_s2)


class ComplexWeibullViewportTimeModelMRV3(DefaultViewportTimeModel):
    def __init__(self, max_vertical_id):
        super(ComplexWeibullViewportTimeModelMRV3, self).__init__()
        self.scale = 1000. # 可以自己调调
        self.epsilon = 0.1 / self.scale
        self.time_threshold = self.convert_v(VPT_EPSILON)
        self.max_vertical_id = 1 # max_vertical_id

        self.s_ep0 = [0. for _ in xrange(self.max_vertical_id)] # k
        self.m_ep0 = [0. for _ in xrange(self.max_vertical_id)] # lambda
        self.s_ep1_e0 = [1.778 for _ in xrange(self.max_vertical_id)] # 0.5
        self.m_ep1_e0 = [0.002 for _ in xrange(self.max_vertical_id)] # 1
        self.s_ep1_e1_r0 = [1.278 for _ in xrange(self.max_vertical_id)] # 2
        self.m_ep1_e1_r0 = [4.875 for _ in xrange(self.max_vertical_id)] # 1
        self.s_ep1_e1_r1_s0 = [1.494 for _ in xrange(self.max_vertical_id)] # 5
        self.m_ep1_e1_r1_s0 = [18.43 for _ in xrange(self.max_vertical_id)] # 1
        self.s_ep1_e1_r1_s1 = [1.2 for _ in xrange(self.max_vertical_id)]
        self.m_ep1_e1_r1_s1 = [-1. for _ in xrange(self.max_vertical_id)]
        self.s_ep1_e1_r1_s2 = [1.2 for _ in xrange(self.max_vertical_id)]
        self.m_ep1_e1_r1_s2 = [-1. for _ in xrange(self.max_vertical_id)]

        self.p_ep0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s1 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s1 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s2 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s2 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_origin = 0

    def get_method_name(self):
        return 'ComplexWeibullViewportTimeModelMRV3'

    def weibull_quality_function(self, x, k, l):
        return k/l * (x/l)**(k-1) * math.exp(-(x/l)**k)

    def convert_v(self, v):
        # v = math.log(max(self.epsilon, v/self.scale))
        v = max(self.epsilon, v / self.scale)
        return v

    def P(self, v, **kwargs):
        # return 1.
        self.v_origin = v
        v = self.convert_v(v)
        layout = 0 # layout = kwargs['layout']
        if 'EP' in kwargs and kwargs['EP'] == 0:
            # if 'C' in kwargs and kwargs['C'] == 1:
            #     return self.epsilon
            # elif v < self.time_threshold:
            #     return 100.
            # else:
            #     s = self.s_e0[layout]
            #     m = self.m_e0[layout]
            #     return max(self.normal_quality_function(v, s, m), self.epsilon)
            if v < self.time_threshold:
                return 100
            else:
                return 1e-12
        elif 'EP' in kwargs and kwargs['EP'] == 1:
            # v = math.log(max(20./self.scale, self.v_origin/self.scale))
            if 'E' in kwargs and kwargs['E'] == 0:
                s = self.s_ep1_e0[layout]
                m = self.m_ep1_e0[layout]
                # return max(self.normal_quality_function(v, s, m), self.epsilon)
            elif 'E' in kwargs and kwargs['E'] == 1:
                if 'R' in kwargs and kwargs['R'] == 1:
                    s = self.s_ep1_e1_r1_s0[layout]
                    m = self.m_ep1_e1_r1_s0[layout]
                    # return max(self.normal_quality_function(v, s, m), self.epsilon)
                else:
                    s = self.s_ep1_e1_r0[layout]
                    m = self.m_ep1_e1_r0[layout]
                    # return max(self.normal_quality_function(v, s, m), self.epsilon)
            else:
                raise ValueError('ComplexWeibullViewportTimeModelMRV3 P() input error!')
            return max(self.weibull_quality_function(v, s, m), self.epsilon)
        else:
            raise ValueError('ComplexWeibullViewportTimeModelMRV3 P() input error!')

    def update_init(self):
        """
        initialize the training parameters before M-step
        Returns: None

        """
        self.p_ep0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s1 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s1 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s2 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s2 = [[] for _ in xrange(self.max_vertical_id)]

    def update(self, posterior, v, **kwargs):
        """
        update the training parameters when there is an item in the Q-function of the form
        P(conditions|C_1..M, V_1..M) * log P(V_i|conditions)
        Args:
            posterior:  posterior prob P(conditions|C_1..M, V_1..M) estimated in E-step
            v: viewport time
            **kwargs: conditions su

        Returns: None

        """
        v = self.convert_v(v)
        if v < self.time_threshold:
            return
        layout = 0 # kwargs['layout']
        posterior = max(self.epsilon, posterior)
        if 'EP' in kwargs and kwargs['EP'] == 1:
            if 'E' in kwargs and kwargs['E'] == 0:
                self.p_ep1_e0[layout].append(posterior)
                self.v_ep1_e0[layout].append(v)
            elif 'E' in kwargs and kwargs['E'] == 1:
                if 'R' in kwargs and kwargs['R'] == 1:
                    self.p_ep1_e1_r1_s0[layout].append(posterior)
                    self.v_ep1_e1_r1_s0[layout].append(v)
                else:
                    self.p_ep1_e1_r0[layout].append(posterior)
                    self.v_ep1_e1_r0[layout].append(v)

    def get_l_from_k(self, p, v, k):
        return np.power(np.sum(p * np.power(v, k)) / np.sum(p), 1. / k)

    def get_k_from_k(self, p, v, k):
        l = self.get_l_from_k(p, v, k)
        k_ = math.log(l)
        k_ += np.sum(p * np.log(v / l) * np.power(v / l, k)) / np.sum(p)
        k_ -= np.sum(p * np.log(v)) / np.sum(p)
        k_ = 1. / k_
        return k_, l

    def updata_s_m(self, e_p, e_v, s_orgin, m_orgin):
        # get average k and t
        min_num = 50
        s_list, m_list = [], []
        avg_s, avg_m = [], []
        for i in xrange(len(e_p)):
            # print len(e_p[i])
            if len(e_p[i]) >= min_num:
                avg_s.append(s_orgin[i])
                avg_m.append(m_orgin[i])
            else:
                raise ValueError('ComplexNormalViewportTimeModelMR updata_s_m() min num < 50')
        avg_s = np.mean(avg_s)
        avg_m = np.mean(avg_m)
        # update s, m
        for i in xrange(len(e_p)):
            if len(e_p[i]) < min_num:
                s_list.append(avg_s)
                m_list.append(avg_m)
                continue
            p = np.array(e_p[i], dtype=np.float64)
            v = np.array(e_v[i], dtype=np.float64)
            # m = np.sum(p*v) / np.sum(p)
            # s = math.sqrt(np.sum(p*np.power(v-m, 2))/np.sum(p))
            m = np.sum(p*np.log(v)) / np.sum(p)
            s = math.sqrt(np.sum(p*np.power(np.log(v)-m, 2))/np.sum(p))

            s = s_orgin[i]
            m = m_orgin[i]
            for _ in range(10):
                if s < self.epsilon:
                    s = self.epsilon
                    m = self.get_l_from_k(p, v, s)
                    # print '!!!'
                    break
                s_ = s
                s, m = self.get_k_from_k(p, v, s)
                if math.isnan(s) or math.isinf(m) or math.isnan(m) or math.isinf(s) or s > 50. or m > 50.:
                    s = avg_s
                    m = avg_m
                    # print '~~~'
                # if math.fabs(k - k_) < 10.: # self.epsilon
                #     break
            s_list.append(s)
            m_list.append(m)
        # print s_list
        # print m_list
        # print ''
        return s_list, m_list

    def update_finalize(self):
        """
        update the model parameters using the training parameters
        Returns: None

        """
        # self.s_ep0, self.m_ep0 = self.updata_s_m(self.p_ep0, self.v_ep0, self.s_ep0, self.m_ep0)
        self.s_ep1_e0, self.m_ep1_e0 = self.updata_s_m(self.p_ep1_e0, self.v_ep1_e0, self.s_ep1_e0, self.m_ep1_e0)
        self.s_ep1_e1_r0, self.m_ep1_e1_r0 = self.updata_s_m(self.p_ep1_e1_r0, self.v_ep1_e1_r0, self.s_ep1_e1_r0, self.m_ep1_e1_r0)
        self.s_ep1_e1_r1_s0, self.m_ep1_e1_r1_s0 = self.updata_s_m(self.p_ep1_e1_r1_s0, self.v_ep1_e1_r1_s0, self.s_ep1_e1_r1_s0, self.m_ep1_e1_r1_s0)
        # self.s_ep1_e1_r1_s1, self.m_ep1_e1_r1_s1 = self.updata_s_m(self.p_ep1_e1_r1_s1, self.v_ep1_e1_r1_s1, self.s_ep1_e1_r1_s1, self.m_ep1_e1_r1_s1)
        # self.s_ep1_e1_r1_s2, self.m_ep1_e1_r1_s2 = self.updata_s_m(self.p_ep1_e1_r1_s2, self.v_ep1_e1_r1_s2, self.s_ep1_e1_r1_s2, self.m_ep1_e1_r1_s2)


class ComplexGammaViewportTimeModelMRV3(DefaultViewportTimeModel):
    def __init__(self, max_vertical_id):
        super(ComplexGammaViewportTimeModelMRV3, self).__init__()
        self.scale = 1000. # 可以自己调调
        self.epsilon = 0.1 / self.scale
        self.time_threshold = self.convert_v(VPT_EPSILON)
        self.max_vertical_id = 1 # max_vertical_id

        self.s_ep0 = [0.75 for _ in xrange(self.max_vertical_id)] # k
        self.m_ep0 = [1. for _ in xrange(self.max_vertical_id)] # theta
        self.s_ep1_e0 = [1. for _ in xrange(self.max_vertical_id)] # 1
        self.m_ep1_e0 = [0.5 for _ in xrange(self.max_vertical_id)] # 2
        self.s_ep1_e1_r0 = [2. for _ in xrange(self.max_vertical_id)] # 2
        self.m_ep1_e1_r0 = [2. for _ in xrange(self.max_vertical_id)] # 2
        self.s_ep1_e1_r1_s0 = [3. for _ in xrange(self.max_vertical_id)] # 3
        self.m_ep1_e1_r1_s0 = [2. for _ in xrange(self.max_vertical_id)] # 2
        self.s_ep1_e1_r1_s1 = [1.2 for _ in xrange(self.max_vertical_id)]
        self.m_ep1_e1_r1_s1 = [-1. for _ in xrange(self.max_vertical_id)]
        self.s_ep1_e1_r1_s2 = [1.2 for _ in xrange(self.max_vertical_id)]
        self.m_ep1_e1_r1_s2 = [-1. for _ in xrange(self.max_vertical_id)]

        self.p_ep0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s1 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s1 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s2 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s2 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_origin = 0

    def get_method_name(self):
        return 'ComplexGammaViewportTimeModelMRV3'

    def gamma_quality_function(self, x, k, t):
        return (x ** (k - 1.)) * math.exp(-x / t) / (math.gamma(k) * (t ** k))

    def convert_v(self, v):
        # v = math.log(max(self.epsilon, v/self.scale))
        v = max(self.epsilon, v / self.scale)
        return v

    def P(self, v, **kwargs):
        # return 1.
        self.v_origin = v
        v = self.convert_v(v)
        layout = 0 # layout = kwargs['layout']
        if 'EP' in kwargs and kwargs['EP'] == 0:
            # if 'C' in kwargs and kwargs['C'] == 1:
            #     return self.epsilon
            # elif v < self.time_threshold:
            #     return 100.
            # else:
            #     s = self.s_e0[layout]
            #     m = self.m_e0[layout]
            #     return max(self.normal_quality_function(v, s, m), self.epsilon)
            if v < self.time_threshold:
                return 100
            else:
                return 1e-12
        elif 'EP' in kwargs and kwargs['EP'] == 1:
            # v = math.log(max(20./self.scale, self.v_origin/self.scale))
            if 'E' in kwargs and kwargs['E'] == 0:
                s = self.s_ep1_e0[layout]
                m = self.m_ep1_e0[layout]
                # return max(self.normal_quality_function(v, s, m), self.epsilon)
            elif 'E' in kwargs and kwargs['E'] == 1:
                if 'R' in kwargs and kwargs['R'] == 1:
                    s = self.s_ep1_e1_r1_s0[layout]
                    m = self.m_ep1_e1_r1_s0[layout]
                    # return max(self.normal_quality_function(v, s, m), self.epsilon)
                else:
                    s = self.s_ep1_e1_r0[layout]
                    m = self.m_ep1_e1_r0[layout]
                    # return max(self.normal_quality_function(v, s, m), self.epsilon)
            else:
                raise ValueError('ComplexGammaViewportTimeModelMRV3 P() input error!')
            return max(self.gamma_quality_function(v, s, m), self.epsilon)
        else:
            raise ValueError('ComplexGammaViewportTimeModelMRV3 P() input error!')

    def update_init(self):
        """
        initialize the training parameters before M-step
        Returns: None

        """
        self.p_ep0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s1 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s1 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_ep1_e1_r1_s2 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_ep1_e1_r1_s2 = [[] for _ in xrange(self.max_vertical_id)]

    def update(self, posterior, v, **kwargs):
        """
        update the training parameters when there is an item in the Q-function of the form
        P(conditions|C_1..M, V_1..M) * log P(V_i|conditions)
        Args:
            posterior:  posterior prob P(conditions|C_1..M, V_1..M) estimated in E-step
            v: viewport time
            **kwargs: conditions su

        Returns: None

        """
        v = self.convert_v(v)
        if v < self.time_threshold:
            return
        layout = 0 # kwargs['layout']
        posterior = max(self.epsilon, posterior)
        if 'EP' in kwargs and kwargs['EP'] == 1:
            if 'E' in kwargs and kwargs['E'] == 0:
                self.p_ep1_e0[layout].append(posterior)
                self.v_ep1_e0[layout].append(v)
            elif 'E' in kwargs and kwargs['E'] == 1:
                if 'R' in kwargs and kwargs['R'] == 1:
                    self.p_ep1_e1_r1_s0[layout].append(posterior)
                    self.v_ep1_e1_r1_s0[layout].append(v)
                else:
                    self.p_ep1_e1_r0[layout].append(posterior)
                    self.v_ep1_e1_r0[layout].append(v)

    def psi(self, x):
        delta = self.epsilon ** 2 * 1e-2
        y = math.gamma(x)
        y0 = math.gamma(x - delta)
        y1 = math.gamma(x + delta)
        return (y1 - y0) / (2. * delta * y)

    def dpsi(self, x):
        delta = self.epsilon ** 2
        return (self.psi(x + delta) - self.psi(x - delta)) / (2. * delta)

    def updata_s_m(self, e_p, e_v, s_orgin, m_orgin):
        # get average k and t
        min_num = 50
        s_list, m_list = [], []
        avg_s, avg_m = [], []
        for i in xrange(len(e_p)):
            # print len(e_p[i])
            if len(e_p[i]) >= min_num:
                avg_s.append(s_orgin[i])
                avg_m.append(m_orgin[i])
            else:
                raise ValueError('ComplexGammaViewportTimeModelMRV3 updata_s_m() min num < 50')
        avg_s = np.mean(avg_s)
        avg_m = np.mean(avg_m)
        # update s, m
        for i in xrange(len(e_p)):
            if len(e_p[i]) < min_num:
                s_list.append(avg_s)
                m_list.append(avg_m)
                continue
            p = np.array(e_p[i], dtype=np.float64)
            v = np.array(e_v[i], dtype=np.float64)

            s = np.log(np.sum(p * v) / np.sum(p))
            s -= np.sum(p * np.log(v)) / np.sum(p)
            if math.fabs(s) < self.epsilon:
                s_list.append(avg_s)
                m_list.append(avg_m)
                # print '~~~'
                continue
            k = (3 - s + math.sqrt((s - 3.) ** 2 + 24 * s)) / (12 * s)
            t = 0.
            while True:
                if k < self.epsilon:
                    k = self.epsilon
                    t = np.sum(p * v) / (k * np.sum(p))
                    # print '!!!'
                    break
                if k > 50.:
                    k = avg_s
                    t = avg_m
                    # print '***'
                    break
                k_ = k
                k -= (math.log(k) - self.psi(k) - s) / (1. / k - self.dpsi(k))
                if math.fabs(k - k_) < self.epsilon:
                    t = np.sum(p * v) / (k * np.sum(p))
                    break
            s_list.append(k)
            m_list.append(t)
            # print s_list
            # print m_list
            # print ''
        return s_list, m_list



    def update_finalize(self):
        """
        update the model parameters using the training parameters
        Returns: None

        """
        # self.s_ep0, self.m_ep0 = self.updata_s_m(self.p_ep0, self.v_ep0, self.s_ep0, self.m_ep0)
        self.s_ep1_e0, self.m_ep1_e0 = self.updata_s_m(self.p_ep1_e0, self.v_ep1_e0, self.s_ep1_e0, self.m_ep1_e0)
        self.s_ep1_e1_r0, self.m_ep1_e1_r0 = self.updata_s_m(self.p_ep1_e1_r0, self.v_ep1_e1_r0, self.s_ep1_e1_r0, self.m_ep1_e1_r0)
        self.s_ep1_e1_r1_s0, self.m_ep1_e1_r1_s0 = self.updata_s_m(self.p_ep1_e1_r1_s0, self.v_ep1_e1_r1_s0, self.s_ep1_e1_r1_s0, self.m_ep1_e1_r1_s0)
        # self.s_ep1_e1_r1_s1, self.m_ep1_e1_r1_s1 = self.updata_s_m(self.p_ep1_e1_r1_s1, self.v_ep1_e1_r1_s1, self.s_ep1_e1_r1_s1, self.m_ep1_e1_r1_s1)
        # self.s_ep1_e1_r1_s2, self.m_ep1_e1_r1_s2 = self.updata_s_m(self.p_ep1_e1_r1_s2, self.v_ep1_e1_r1_s2, self.s_ep1_e1_r1_s2, self.m_ep1_e1_r1_s2)


