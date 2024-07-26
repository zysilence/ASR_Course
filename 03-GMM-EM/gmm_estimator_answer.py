# Author: Sining Sun , Zhanheng Yang

import numpy as np
from utils import *
import scipy.cluster.vq as vq

num_gaussian = 5
num_iterations = 5
targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

class GMM:
    def __init__(self, D, K=5):
        assert(D>0)
        self.dim = D
        self.K = K
        #Kmeans Initial
        self.mu , self.sigma , self.pi = self.kmeans_initial()

    def kmeans_initial(self):
        mu = []
        sigma = []
        data = read_all_data('train/feats.scp')
        (centroids, labels) = vq.kmeans2(data, self.K, minit="points", iter=100)
        clusters = [[] for i in range(self.K)]
        for (l,d) in zip(labels,data):
            clusters[l].append(d)

        for cluster in clusters:
            mu.append(np.mean(cluster, axis=0))
            sigma.append(np.cov(cluster, rowvar=0))
        pi = np.array([len(c)*1.0 / len(data) for c in clusters])
        return mu , sigma , pi
    
    def gaussian(self , x , mu , sigma):
        """Calculate gaussion probability.
    
            :param x: The observed data, dim*1.
            :param mu: The mean vector of gaussian, dim*1
            :param sigma: The covariance matrix, dim*dim
            :return: the gaussion probability, scalor
        """
        D=x.shape[0]
        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma + 0.0001)
        mahalanobis = np.dot(np.transpose(x-mu), inv_sigma)
        mahalanobis = np.dot(mahalanobis, (x-mu))
        const = 1/((2*np.pi)**(D/2))
        return const * (det_sigma)**(-0.5) * np.exp(-0.5 * mahalanobis)
    
    def calc_log_likelihood(self, X):
        """Calculate log likelihood of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of current model
        """

        log_llh = 0.0
        N = X.shape[0]
        for n in range(N):
            tmp = 0.0
            for k in range(self.K):
                tmp += self.pi[k] * self.gaussian(X[n], self.mu[k], self.sigma[k])
            log_llh += np.log(tmp)
        return log_llh

    def em_estimator(self, X):
        """Update paramters of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of updated model
        """

        log_llh = 0.0
        # X的shape为（1937，39）
        # 每一个小x的shape为（39，）
        # 首先构建一个（1937，5）的二维矩阵用于存放后验概率
        N = X.shape[0]
        gama = np.zeros((N, self.K))

        # 先遍历计算每一个后验概率的分子
        for n in range(N):
            for k in range(self.K):
                gama[n][k] = self.pi[k] * self.gaussian(X[n], self.mu[k], self.sigma[k])
        # 列相加计算对应的分母
        tmp = np.sum(gama, axis=1)
        for n in range(N):
            # 分子除分母
            gama[n] /= tmp[n]

        # 根据计算的后验概率重新估计参数
        # 首先计算Nk
        Nk = np.sum(gama, axis=0)

        # 更新pi,pi[k]所有元素之和应该为1
        self.pi = Nk/N

        # 更新mu,单个mu[k]的shape为（39，）
        self.mu = list()
        for k in range(self.K):
            tmp = np.zeros(self.dim)
            for n in range(N):
                tmp += X[n]*gama[n][k]
            tmp /= Nk[k]
            self.mu.append(tmp)

        # 更新sigma[k]  sigma[k]的shape为(39,39)
        self.sigma = list()
        for k in range(self.K):
            tmp = np.zeros((self.dim, self.dim))
            for n in range(N):
                tmp += gama[n][k] * np.outer(X[n]-self.mu[k], X[n]-self.mu[k])
            tmp /= Nk[k]
            self.sigma.append(tmp)

        # 计算似然估计
        log_llh = self.calc_log_likelihood(X)
        return log_llh


def train(gmms, num_iterations = num_iterations):
    dict_utt2feat, dict_target2utt = read_feats_and_targets('train/feats.scp', 'train/text')
    
    for target in targets:
        feats = get_feats(target, dict_utt2feat, dict_target2utt)   #
        for i in range(num_iterations):
            log_llh = gmms[target].em_estimator(feats)
    return gmms

def test(gmms):
    correction_num = 0
    error_num = 0
    acc = 0.0
    dict_utt2feat, dict_target2utt = read_feats_and_targets('test/feats.scp', 'test/text')
    dict_utt2target = {}
    for target in targets:
        utts = dict_target2utt[target]
        for utt in utts:
            dict_utt2target[utt] = target
    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])
        scores = []
        for target in targets:
            scores.append(gmms[target].calc_log_likelihood(feats))
        predict_target = targets[scores.index(max(scores))]
        if predict_target == dict_utt2target[utt]:
            correction_num += 1
        else:
            error_num += 1
    acc = correction_num * 1.0 / (correction_num + error_num)
    return acc


def main():
    gmms = {}
    for target in targets:
        gmms[target] = GMM(39, K=num_gaussian) #Initial model
    gmms = train(gmms)
    acc = test(gmms)
    print('Recognition accuracy: %f' % acc)
    fid = open('acc.txt', 'w')
    fid.write(str(acc))
    fid.close()


if __name__ == '__main__':
    main()
