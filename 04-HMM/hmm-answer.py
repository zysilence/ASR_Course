# Author: Kaituo Xu, Fan Yu
import numpy as np


# 前向算法
def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    # 观测序列的长度记为T
    T = len(O)
    # 状态共有N种
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # Put Your Code Here

    # 第一步，初始化alpha并计算其初值
    alpha = np.zeros((T, N))
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][O[0]]
    # 第二步，开始递推，直到达到终止条件,复杂度为O(T*N^2)
    for t in range(1, T):
        for i in range(N):
            temp = 0.0
            for j in range(N):
                temp += alpha[t - 1][j] * A[j][i]
            alpha[t][i] = temp * B[i][O[t]]
    # 求和
    for i in range(N):
        prob += alpha[T - 1][i]
    # End Assignment
    return prob


# 后向算法
def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # Put Your Code Here
    # 第一步，初始化beta并计算其初值
    beta = np.zeros((T, N))
    for i in range(N):
        beta[T - 1][i] = 1
    # 第二步，开始递推，直到达到终止条件,复杂度也为O(T*N^2)
    for t in range(T - 2, -1, -1):
        for i in range(N):
            temp = 0.0
            for j in range(N):
                temp += A[i][j] * B[j][O[t + 1]] * beta[t + 1][j]
            beta[t][i] = temp
    # 求和
    for i in range(N):
        prob += pi[i] * B[i][O[0]] * beta[0][i]
    # End Assignment
    return prob


# 维特比算法
def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob, best_path = 0.0, []
    # Begin Assignment

    # Put Your Code Here
    # 第一步初始化
    A_max = np.zeros((T, N))
    Path = np.zeros((T, N), dtype=np.int32)
    for i in range(N):
        A_max[0][i] = pi[i] * B[i][O[0]]
        Path[0][i] = -1

    # 第二步，开始递推,直到达到终止条件
    for t in range(1, T):
        for i in range(N):
            temp = np.zeros(N)
            for j in range(N):
                temp[j] = A_max[t - 1][j] * A[j][i]
            A_max[t][i] = temp.max() * B[i][O[t]]
            Path[t][i] = temp.argmax()
    # 对数组进行数值填充，从而可以通过索引改变值
    for t in range(T):
        best_path.append(-1)
    # 筛选出最大的概率
    best_prob = A_max[T - 1].max()
    # 筛选出对应的It*
    best_path[T - 1] = A_max[T - 1].argmax()
    # 最优路径回溯
    for t in range(T - 2, -1, -1):
        best_path[t] = Path[t + 1][best_path[t + 1]]

    # 最优回溯完成后，要将best_path存储的状态下标加1转换成状态。
    for t in range(T):
        best_path[t] += 1
        # 转换为（时刻，状态）进行输出
        best_path[t] = tuple((t+1, best_path[t]))
    # End Assignment

    return best_prob, best_path


if __name__ == "__main__":
    # 状态集合= {1, 2, 3} ，观测集合= {红，白} N=3,M=2
    color2id = {"RED": 0, "WHITE": 1}
    # model parameters
    # 初始状态概率向量
    pi = [0.2, 0.4, 0.4]
    # 状态转移概率矩阵（N,N）
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    # 观测概率矩阵 (N,M)
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (0, 1, 0)
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward)

    best_prob, best_path = Viterbi_algorithm(observations, HMM_model)
    print(best_prob, best_path)

