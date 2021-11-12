import struct
import matplotlib.pyplot as plt
import numpy as np 
import math
import scipy.stats as stats
import sys
from ConfidenceEllipse import ConfidenceEllipse



#バイナリファイル読み込み用関数
def read_binaryfloat(filename):
    f=open(filename, 'rb')
    x = []
    while True:
        #4yteごとfloatで読み込む
        temp_x = f.read(4)
        temp_y = f.read(4)
        if not temp_y:
            break
        x.append([struct.unpack('f',temp_x)[0],struct.unpack('f',temp_y)[0]])
    return np.array(x)


def gausian(u,s,x):
    res=math.exp(-1/2*np.dot((x-u).T,np.dot(np.linalg.inv(s),x-u)))/(math.pow(2*math.pi,len(u)/2)*math.sqrt(np.linalg.det(s)))
    return res


#平均ベクトル計算用
def get_average_vector (x,y):
    ave_x,ave_y = 0,0
    for i,j in zip(x,y):
        ave_x += i
        ave_y += j
    ave_x /= len(x)
    ave_y /= len(y)
    return (ave_x,ave_y)

#対数尤度
def log_likelihood(u,s,w,x):
    sum = 0.0
    for i,j,k in zip(u,s,w):
        sum += k*gausian(i,j,x)
    return math.log(sum)

#EMアルゴリズムに基づき混合正規分布を求める
def fit_mix_normaldistribution(x,distribution_num):
    dim = len(x[0])
    likelihood_list = []
    #初期値の設定
    u = np.zeros((distribution_num,2))
    for i in range(distribution_num):
        u[i] += i
    s = np.array([np.eye(2) for i in range(distribution_num)])
    w = np.ones(distribution_num)/distribution_num
    responsibility = np.zeros((len(x),distribution_num))
    
    cnt = 1
    prev = 0
    for n in range(len(x)):
            prev += log_likelihood(u,s,w,x[n])

    for i in range(200):
        #Eステップ
        for n in range(len(x)):
            denominator = 0
            for l in range(distribution_num):
                denominator += w[l] * gausian(u[l],s[l],x[n])
            for k in range(distribution_num):
                responsibility[n][k] = w[k]*gausian(u[k],s[k],x[n])/denominator
    
        
        #Mステップ(更新)
        # print(responsibility)
        N = np.array([np.sum(responsibility[:,k]) for k in range(distribution_num)])
        for k in range(distribution_num):
            #共分散行列の更新
            temp = np.zeros(s[k].shape)
            for n in range(len(x)):
                temp += responsibility[n][k]*np.dot((x[n]-u[k]).reshape(1,dim).T,(x[n]-u[k]).reshape(1,dim))
            s[k] = 1/N[k]*temp
            #u_newの更新
            temp = np.zeros(dim)
            for n in range(len(x)):
                temp += responsibility[n][k]*x[n]
            
            u[k] = 1/N[k]*temp
            #重みの更新
            w[k] = N[k]/len(x)
        
        #対数尤度の計算
        log_prob = 0
        for n in range(len(x)):
            log_prob += log_likelihood(u,s,w,x[n])
        print(cnt,':',log_prob)
        if abs(log_prob-prev) < 0.01:
            break
        cnt +=1
        likelihood_list.append(log_prob)
        prev=log_prob

    return u,s,w,likelihood_list

def calc_acuracy(u1,u2,s1,s2,w1,w2,x1,x2):
    correct = 0
    for i in x1:
        if log_likelihood(u1,s1,w1,i) > log_likelihood(u2,s2,w2,i):
            correct += 1
    for i in x2:
        if log_likelihood(u1,s1,w1,i) < log_likelihood(u2,s2,w2,i):
            correct += 1
    print(correct,'/',len(x1)+len(x2))
    print('accuracy=',correct/(len(x1)+len(x2))*100)



if __name__ == '__main__':
    if len(sys.argv)>1:
        distribution_num = int(sys.argv[1])
        print(sys.argv[1])
    #デフォルトは分布二つの混合正規分布
    else:
        distribution_num = 2

    x1 = read_binaryfloat('data/class1.dat')
    x2  = read_binaryfloat('data/class2.dat')
    #散布図でプロット
    fig = plt.figure()
    ax= fig.add_subplot(111)
    plt.scatter(x1[:,0],x1[:,1],label = 'class1',marker='+')
    plt.scatter(x2[:,0],x2[:,1],label = 'class2',marker='*')
    u1,s1,w1,l1 = fit_mix_normaldistribution(x1,distribution_num)
    u2,s2,w2,l2= fit_mix_normaldistribution(x2,distribution_num)
    for i in u1:
        plt.scatter(i[0],i[1],marker='o',color = 'black')
    for i in u2:
        plt.scatter(i[0],i[1],marker='o',color = 'black')
    

    #95%区間のプロット
    for i in range(len(u1)):
        el1 = ConfidenceEllipse(u1[i],s1[i], p=0.95)
        el2 = ConfidenceEllipse(u2[i],s2[i], p=0.95)
        if distribution_num==1:
            ax.add_artist(el1.get_patch(face_color="blue", alpha=0.5))
            ax.add_artist(el2.get_patch(face_color="orange", alpha=0.5))
        else:
            ax.add_artist(el1.get_patch(face_color="blue", alpha=w1[i]))
            ax.add_artist(el2.get_patch(face_color="orange", alpha=w2[i]))

    plt.xlabel('1st Order')
    plt.ylabel('2nd Order')
    plt.legend()
    plt.show()

    plt.plot(l1,label='class1')
    plt.plot(l2,label='class2')
    plt.xlabel('iteration num')
    plt.ylabel('sum of log_likelihood')
    plt.legend()
    plt.show()

    #精度計算
    calc_acuracy(u1, u2,s1,s2,w1,w2,x1,x2)


