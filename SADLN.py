import argparse
import sys
import numpy as np
import random
import time
import os

import tensorflow as tf
from subprocess import check_output
import h5py
import re
import math
import pandas as pd
from os.path import splitext, basename, isfile
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn import mixture
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, BatchNormalization, Dropout, Activation, merge, Conv2D, \
    MaxPooling2D, Activation, LeakyReLU, concatenate, Embedding
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from itertools import combinations
import bisect
from transformer import Attention,Position_Embedding
import matplotlib.pyplot as plt


seed = 2
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class ConsensusCluster:
    def __init__(self, cluster, L, K, H, resample_proportion=0.8):
        self.cluster_ = cluster
        self.resample_proportion_ = resample_proportion
        self.L_ = L
        self.K_ = K
        self.H_ = H
        self.Mk = None
        self.Ak = None
        self.deltaK = None
        self.bestK = None

    def _internal_resample(self, data, proportion):
        ids = np.random.choice(
            range(data.shape[0]), size=int(data.shape[0] * proportion), replace=False)
        return ids, data[ids, :]

    def fit(self, data):
        Mk = np.zeros((self.K_ - self.L_, data.shape[0], data.shape[0]))
        Is = np.zeros((data.shape[0],) * 2)
        for k in range(self.L_, self.K_):
            i_ = k - self.L_
            for h in range(self.H_):
                ids, dt = self._internal_resample(data, self.resample_proportion_)
                Mh = self.cluster_(n_clusters=k).fit_predict(dt)
                ids_sorted = np.argsort(Mh)
                sorted_ = Mh[ids_sorted]
                for i in range(k):
                    ia = bisect.bisect_left(sorted_, i)
                    ib = bisect.bisect_right(sorted_, i)
                    is_ = ids_sorted[ia:ib]
                    ids_ = np.array(list(combinations(is_, 2))).T
                    if ids_.size != 0:
                        Mk[i_, ids_[0], ids_[1]] += 1
                ids_2 = np.array(list(combinations(ids, 2))).T
                Is[ids_2[0], ids_2[1]] += 1
            Mk[i_] /= Is + 1e-8
            Mk[i_] += Mk[i_].T
            Mk[i_, range(data.shape[0]), range(
                data.shape[0])] = 1
            Is.fill(0)
        self.Mk = Mk
        self.Ak = np.zeros(self.K_ - self.L_)
        for i, m in enumerate(Mk):
            hist, bins = np.histogram(m.ravel(), density=True)
            self.Ak[i] = np.sum(h * (b - a)
                                for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist)))
        self.deltaK = np.array([(Ab - Aa) / Aa if i > 2 else Aa
                                for Ab, Aa, i in zip(self.Ak[1:], self.Ak[:-1], range(self.L_, self.K_ - 1))])
        self.bestK = np.argmax(self.deltaK) + \
                     self.L_ if self.deltaK.size > 0 else self.L_

    def predict(self):
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            1 - self.Mk[self.bestK - self.L_])

    def predict_data(self, data):
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            data)


class GeLU(Activation):
    def __init__(self, activation, **kwargs):
        super(GeLU, self).__init__(activation, **kwargs)
        self.__name__ = 'gelu'


def gelu(x):   #高斯误差线性单位函数
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


get_custom_objects().update({'gelu': GeLU(gelu)})


class AE():
    def __init__(self, X_shape, n_components, epochs):
        self.epochs = epochs
        sample_size = X_shape[0]
        self.batch_size = 64
        sample_size = X_shape[0]
        self.epochs = 600
        self.n_components = n_components
        self.shape = X_shape[1]

    def train(self, X):

        def expand_dim(args):
            # x, axis = args
            x1 = K.expand_dims(args, axis=0)
            return x1
        def squeeze_dim(args):
            # x, axis = args
            x1 = K.squeeze(args, axis=0)
            return x1

        encoding_dim = self.n_components
        original_dim = X.shape[1]

        # n_samples = X.shape[0]
        # n_steps = int(n_samples // self.batch_size)
        # X = X[:n_steps* self.batch_size]

        input = Input(shape=(original_dim,))
        encoded = Dense(encoding_dim)(input)
        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)

        ############################## 调用Transformer模块 得到经过self-attention更新后的特征 ##################################
     #   encoded = Lambda(expand_dim)(encoded)
        #  d_model 需要被num_heads整除
     #   encoded = Attention(d_model=encoding_dim, num_heads=4)([encoded, encoded, encoded])
     #   encoded = Lambda(squeeze_dim)(encoded)
        ##########################################################################################################################
        z = Dense(encoding_dim, activation='relu')(encoded)
        decoded = Dense(encoding_dim, activation='relu')(z)
        output = Dense(original_dim, activation='sigmoid')(decoded)
        ae = Model(input, output)
        encoder = Model(input, z)
        ae_loss = mse(input, output)
        ae.add_loss(ae_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0004)
        ae.compile(optimizer=optimizer)
        print(len(ae.layers))
        print(ae.count_params())
        print("x shape:" ,X.shape)
        ae.fit(X, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        return encoder.predict(X)


class VAE():
    def __init__(self, X_shape, n_components, epochs):
        self.epochs = epochs
        self.batch_size = 64
        sample_size = X_shape[0]
        self.epochs = 600
        self.n_components = n_components
        self.shape = X_shape[1]

    def train(self, X):
        def sampling(args):
            z_mean, z_log_var = args   # z-means高斯分布的均值    z-log-var 方差的对数
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim), seed=0)   #标准高斯分布
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        def expand_dim(args):
            # x, axis = args
            x1 = K.expand_dims(args, axis=0)
            return x1
        def squeeze_dim(args):
            # x, axis = args
            x1 = K.squeeze(args, axis=0)
            return x1
        encoding_dim = self.n_components
        original_dim = X.shape[1]
        input = Input(shape=(original_dim,))
        encoded = Dense(encoding_dim)(input)
        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)
        ############################## 调用Transformer模块 得到经过self-attention更新后的特征 ##################################
        # encoded = Lambda(expand_dim)(encoded)
        # #  d_model 需要被num_heads整除
        # encoded = Attention(d_model=encoding_dim, num_heads=4)([encoded, encoded, encoded])
        # encoded = Lambda(squeeze_dim)(encoded)
        ##########################################################################################################################
        z_mean = Dense(encoding_dim)(encoded)
        z_log_var = Dense(encoding_dim)(encoded)
        z = Lambda(sampling, output_shape=(encoding_dim,), name='z')([z_mean, z_log_var])
        decoded = Dense(encoding_dim, activation='relu')(z)
        output = Dense(original_dim, activation='sigmoid')(decoded)
        vae = Model(input, output)
        encoder = Model(input, z)
        reconstruction_loss = mse(input, output)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)
        vae.compile(optimizer=optimizer)
        print(len(vae.layers))
        print(vae.count_params())
        vae.fit(X, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        return encoder.predict(X)



class SADLN():
    def __init__(self, datasets, n_latent_dim,weight=0.001, model_path='SADLN.h5', epochs=600, batch_size=64, learning_rate=0.0001):
        self.latent_dim = n_latent_dim
        # optimizer = Adam()
        # TODO 将keras自带的优化器替换为tf的优化器，解决随机性问题
        optimizer =tf.train.AdamOptimizer(learning_rate=0.0001)  # 调参， 更换优化器 Adam(), AdadeltaOptimizer(),
        self.n = len(datasets)
        self.epochs = epochs
        self.batch_size = batch_size
        sample_size = 0
        if self.n > 1:
            sample_size = datasets[0].shape[0]
        print(sample_size)
        self.shape = []
        self.weight = [0.25, 0.25, 0.25, 0.25]    #权重
        self.disc_w = 1e-4
        self.model_path = model_path
        input = []
        loss = []
        loss_weights = []
        output = []
        for i in range(self.n):
            self.shape.append(datasets[i].shape[1]) # 每个数据集的属性个数 [3139, 3217, 3105, 1024]
            loss.append('mse')
        loss.append('binary_crossentropy')
        self.decoder, self.disc = self.build_decoder_disc()
        self.disc.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.encoder = self.build_encoder()
        self.encoder.summary()
        for i in range(self.n):
            input.append(Input(shape=(self.shape[i],)))
            loss_weights.append((1 - self.disc_w) * self.weight[i])
        loss_weights.append(self.disc_w)
        z_mean, z_log_var, z = self.encoder(input) # 四种数据通过encoer得到维度为100的特征

        output = self.decoder(z) # 100的特征z通过decoder得到四种数据对应维度大小的特征
        self.gan = Model(input, output)
        self.gan.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)
        print(self.gan.summary())
        return

    def build_encoder(self):
        def sampling(args):
            z_mean, z_log_var = args
            return z_mean + K.exp(0.5 * z_log_var) * K.random_normal(K.shape(z_mean), seed=0)

        def expand_dim(args):
            # x, axis = args
            x1 = K.expand_dims(args, axis=0)
            return x1
        def squeeze_dim(args):
            # x, axis = args
            x1 = K.squeeze(args, axis=0)
            return x1

        encoding_dim = self.latent_dim
        X = []
        dims = []
        denses = []
        denses_Together=[]
        for i in range(self.n):
            X.append(Input(shape=(self.shape[i],)))
            dims.append(int(encoding_dim * self.weight[i]))  # weight  [0.3, 0.1, 0.1, 0.5]
        for i in range(self.n):
            denses = Dense(dims[i])(X[i])
################################################################################################
            denses = Lambda(expand_dim)(denses)
            #  d_model 需要被num_heads整除
            denses = Attention(d_model=150, num_heads=1)([denses, denses, denses])
            denses = Lambda(squeeze_dim)(denses)
            denses_Together.append(denses)
################################################################################################
        if self.n > 1:
            merged_dense = concatenate(denses_Together, axis=-1)  # 将四种数据拼接
        else:
            merged_dense = denses_Together[0]
        model = BatchNormalization()(merged_dense)
        model = Activation('gelu')(model)

        ############################## 调用Transformer模块 得到经过self-attention更新后的特征 ##################################
        #model = Lambda(expand_dim)(model)
        #  d_model 需要被num_heads整除
        #model = Attention(d_model=encoding_dim, num_heads=1)([model, model, model])
        #model = Lambda(squeeze_dim)(model)
        ##########################################################################################################################
        model = Dense(encoding_dim)(model)
        z_mean = Dense(encoding_dim)(model)
        z_log_var = Dense(encoding_dim)(model)
        z = Lambda(sampling, output_shape=(encoding_dim,), name='z')([z_mean, z_log_var])

        return Model(X, [z_mean, z_log_var, z])

    def build_decoder_disc(self):
        denses = []
        X = Input(shape=(self.latent_dim,))
        model = Dense(self.latent_dim)(X)
        model = BatchNormalization()(model)
        model = Activation('gelu')(model)
        for i in range(self.n):
            denses.append(Dense(self.shape[i])(model))
        dec = Dense(1, activation='sigmoid')(model)
        denses.append(dec)
        m_decoder = Model(X, denses)
        m_disc = Model(X, dec)
        return m_decoder, m_disc

    def build_disc(self):
        X = Input(shape=(self.latent_dim,))
        dec = Dense(1, activation='sigmoid', kernel_initializer="glorot_normal")(X)
        output = Model(X, dec)
        return output

    def train(self, X_train, bTrain=True, log_interval= 10):
        model_path = self.model_path
        log_file = "./run.log"
        fp = open(log_file, 'w')
        if bTrain:
            # GAN
            valid = np.ones((self.batch_size, 1))
            fake = np.zeros((self.batch_size, 1))
            for epoch in range(self.epochs):
                #  Train Discriminator
                data = []
                idx = np.random.randint(0, X_train[0].shape[0], self.batch_size)
                for i in range(self.n):
                    data.append(X_train[i][idx])
                latent_fake = self.encoder.predict(data)[2]
                latent_real = np.random.normal(size=(self.batch_size, self.latent_dim))
                # 训练生成器，使得encoder的输出latent code z能够骗过dis判别器
                # 原理是使得encoder的输出z的label为0， 随机生成的label为1,并且用bce loss来监督
                d_loss_real = self.disc.train_on_batch(latent_real, valid)
                d_loss_fake = self.disc.train_on_batch(latent_fake, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                outs = data + [valid]
                #  Train Encoder_GAN
                # 1. 训练判别器，使得通过encoder生成的latent code能够被判别为真，1个bce loss
                # 2. 使得deocder生成的x逼近原始输入数据  4个mse loss
                g_loss = self.gan.train_on_batch(data, outs)
                if epoch % log_interval == 0:
                    print(f'Epoch: {epoch} d_loss: {d_loss} g_loss: {g_loss}')
            fp.close()
            self.encoder.save(model_path)
        else:
            self.encoder = load_model(model_path)
        mat = self.encoder.predict(X_train)[0]
        return mat


class SADLN_API(object):
    def __init__(self, model_path='./model/', epochs=600, weight=0.001,batch_size=64, learning_rate=0.0001):
        self.model_path = model_path
        self.score_path = './score/'
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight = weight
        self.learning_rate = learning_rate

    # feature extract
    def feature_gan(self, datasets, index=None, n_components=600, b_decomposition=True, bTrain= True, weight=0.0002):
        if b_decomposition:
            X = self.encoder_gan(datasets, n_components,bTrain=bTrain)
            fea = pd.DataFrame(data=X, index=index, columns=map(lambda x: 'v' + str(x), range(X.shape[1])))
        else:
            fea = np.concatenate(datasets)
        print("feature extract finished!")
        return fea

    def feature_vae(self, df_ori, n_components=100, b_decomposition=True):
        if b_decomposition:
            X = self.encoder_vae(df_ori, n_components)
            print(X)
            fea = pd.DataFrame(data=X, index=df_ori.index,
                               columns=map(lambda x: 'v' + str(x), range(X.shape[1])))
        else:
            fea = df_ori.copy()
        print("feature extract finished!")
        return fea

    def feature_ae(self, df_ori, n_components=100, b_decomposition=True):
        if b_decomposition:
            n_samples = df_ori.shape[0] # 解决数据集不被bs整除的问题
            n_steps = int(n_samples // self.batch_size)
            df_ori = df_ori[:n_steps * self.batch_size]
            X = self.encoder_ae(df_ori, n_components)
            print(X)
            fea = pd.DataFrame(data=X, index=df_ori.index,
                               columns=map(lambda x: 'v' + str(x), range(X.shape[1])))
        else:
            fea = df_ori.copy()
        print("feature extract finished!")
        return fea

    def impute(self, X):
        X.fillna(X.mean())
        return X

    def encoder_gan(self, ldata, n_components=600,bTrain=True):
        # learning_rate=0.001, weight=0.001, model_path='SADLN.h5', epochs=100, batch_size=64
        egan = SADLN(ldata, n_components, self.weight, self.model_path, self.epochs, self.batch_size, self.learning_rate)

        return egan.train(ldata, bTrain=bTrain)

    def encoder_vae(self, df, n_components=100):
        vae = VAE(df.shape, n_components, self.epochs)
        return vae.train(df)

    def encoder_ae(self, df, n_components=100):
        ae = AE(df.shape, n_components, self.epochs)
        return ae.train(df)

    def tsne(self, X):
        model = TSNE(n_components=2)
        return model.fit_transform(X)

    def pca(self, X):
        fea_model = PCA(n_components=200)
        return fea_model.fit_transform(X)

    def gmm(self, n_clusters=28):
        model = mixture.GaussianMixture(n_components=n_clusters, covariance_type='diag')
        return model

    def kmeans(self, n_clusters=28):
        model = KMeans(n_clusters=n_clusters, random_state=0)
        return model

    def spectral(self, n_clusters=28):
        model = SpectralClustering(n_clusters=n_clusters, random_state=0)
        return model

    def hierarchical(self, n_clusters=28):
        model = AgglomerativeClustering(n_clusters=n_clusters)
        return model


def main(argv=sys.argv):
    # 参数设置
    parser = argparse.ArgumentParser(description='SADLN v1.0')         #创建解析器
    parser.add_argument("-i", dest='file_input', default="./input/input.list",   #添加参数
                        help="file input")
    parser.add_argument("-e", dest='epochs', type=int, default=200, help="Number of iterations") #迭代次数
    parser.add_argument("-m", dest='run_mode', default="feature", help="run_mode: feature, cluster") #运行模式
    parser.add_argument("-w", dest='disc_weight', type=float, default=1e-4, help="weight")
    parser.add_argument("-o", dest='output_path', default="./score/", help="file output")
    parser.add_argument("-p", dest='other_approach', default="spectral", help="kmeans, spectral, tsne_gmm, tsne")
    parser.add_argument("-s", dest='surv_path',
                        default="./data/TCGA/clinical_PANCAN_patient_with_followup.tsv",
                        help="surv input")
    parser.add_argument("-t", dest='type', default="ALL", help="cancer type: BRCA, GBM")
    parser.add_argument("-cn", dest='cluster_num', default=-1,  help="surv input")
    args = parser.parse_args()


    model_path = './model/' + args.type + '.h5'
    # batch_size  8， 16， 32， 64，
    # learning_rate 学习率 尝试几组参数 1e-2, 1e-3, 2e-3, 1e-4, 2e-4, 3e-4, 1e-5     # lr 和 bs ,bs增大时一般lr也需增加1/2， 找到最优bs和lr搭配时
    # 优化器 Adam(), AdadeltaOptimizer() lr设置为1,
    # bs200, lr 1e-4,     bs400, lr 5e-4,
    SADLN = SADLN_API(model_path, epochs=args.epochs, weight=args.disc_weight, batch_size=64, learning_rate=0.0001)
    cancer_dict = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4,
                   'GBM': 3, 'LUAD': 3, 'PAAD': 2,
                   'SKCM': 4, 'STAD': 3, 'UCEC': 4, 'UVM': 4}
    if args.run_mode == 'SADLN':
        cancer_type = args.type
        if cancer_type not in cancer_dict and args.cluster_num == -1:
            print("Please set the number of clusters!")
        elif args.cluster_num == -1:
            args.cluster_num = cancer_dict[cancer_type]
        fea_tmp_file = './fea/' + cancer_type + '.fea'
        tmp_dir = './fea/' + cancer_type + '/'
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        ldata = []
        l = []
        nb_line = 0
        for line in open(args.file_input, 'rt'): # 遍历fea/cancer_type 文件夹
            base_file = splitext(basename(line.rstrip()))[0]
            fea_save_file = tmp_dir + base_file + '.fea'   #  设置保存文件路径
            if isfile(fea_save_file):
                df_new = pd.read_csv(fea_save_file, sep=',', header=0, index_col=0) # 读取具体文件
                l = list(df_new) # 属性列表
            else:
                clinic_parms = ['bcr_patient_barcode', 'acronym', 'vital_status', 'days_to_death',
                                'days_to_last_followup', 'gender', 'age_at_initial_pathologic_diagnosis',
                                'pathologic_M',
                                'pathologic_N', 'pathologic_T', 'pathologic_stage']
                df = pd.read_csv(args.surv_path, header=0, sep='\t',
                                 usecols=clinic_parms)
                df['status'] = np.where(df['vital_status'] == 'Dead', 1, 0)
                df['days'] = df.apply(lambda r: r['days_to_death'] if r['status'] == 1 else r['days_to_last_followup'],
                                      axis=1)
                df.index = df['bcr_patient_barcode']

                if cancer_type == 'ALL':
                    pass
                else:
                    df = df.loc[df['acronym'] == cancer_type, ::]
                clic_save_file = './results/' + cancer_type + '.clinic'
                df_new = pd.read_csv(line.rstrip(), sep='\t', header=0, index_col=0, comment='#')
                nb_line += 1
                if nb_line == 1:
                    ids = list(df.index)
                    ids_sub = list(df_new)
                    l = list(set(ids) & set(ids_sub))
                    df_clic = df.loc[
                        l, ['status', 'days', 'gender', 'age_at_initial_pathologic_diagnosis', 'pathologic_M',
                            'pathologic_N', 'pathologic_T', 'pathologic_stage']]

                    df_clic.to_csv(clic_save_file, index=True, header=True, sep=',')
                df_new = df_new.loc[::, l]
                df_new = df_new.fillna(0)
                if 'miRNA' in base_file or 'rna' in base_file:
                    df_new = np.log2(df_new + 1)
                scaler = preprocessing.StandardScaler() #在线标准化
                mat = scaler.fit_transform(df_new.values.astype(float))
                df_new.iloc[::, ::] = mat
                print(df_new.shape)
                df_new.to_csv(fea_save_file, index=True, header=True, sep=',')
            df_new = df_new.T #转置 维度：（ n_samples, n_atttribute）
            ldata.append(df_new.values.astype(float)) # 将所有数据合并到ldata中
        start_time = time.time()
        # 提取数据的latent code 。通过对gan进行训练，然后用gan中的encoder输出四种类型数据对于的latent code
        # 维度为(属性个数，100)，将其保存到fea/cancer_type.fea文件下，在第二个命令中将被用到
        # n_components用于控制encoder输出的latent code的维度
        vec = SADLN.feature_gan(ldata, index=l, n_components=600,
                                     bTrain= True,
                                     weight=args.disc_weight) # 使用feature_gan提取latent code特征

        df = pd.DataFrame(data=[time.time() - start_time])
        vec.to_csv(fea_tmp_file, header=True, index=True, sep='\t') # 保存方法
        out_file = './results/' + cancer_type + '.SADLN.time'
        df.to_csv(out_file, header=True, index=False, sep=',')
        # 已将数据通过gan转为latent code
###############################################################################################################
        # 基于latent code做了gmm聚类操作
        if isfile(fea_tmp_file):
            X = pd.read_csv(fea_tmp_file, header=0, index_col=0, sep='\t') # 四种类型数据的latent code （col, 100）
            # 调用sklearn中gmm模型进行聚类预测 (col, 101)
            X['SADLN'] = SADLN.gmm(args.cluster_num).fit_predict(X.values) + 1
            X = X.loc[:, ['SADLN']] # (col, 1)
            out_file = './results/' + cancer_type + '.SADLN'
            X.to_csv(out_file, header=True, index=True, sep='\t')
        else:
            print('file does not exist!')

    elif args.run_mode == 'show':
        cancer_type = args.type
        fea_tmp_file = './fea/' + cancer_type + '.fea'
        out_file = './fea/' + cancer_type + '.tsne'
        label_file = './results/' + cancer_type + '.SADLN'
        if isfile(fea_tmp_file):
            df1 = pd.read_csv(fea_tmp_file, header=0, index_col=0, sep='\t')
            mat = df1.values.astype(float)
            labels = SADLN.tsne(mat)
            print(labels.shape)
            df1['x'] = labels[:, 0]
            df1['y'] = labels[:, 1]
            df1 = df1.loc[:, ['x', 'y']]
            df1.to_csv(out_file, header=True, index=True, sep='\t')
            if isfile(label_file):
                df2 = pd.read_csv(label_file, header=0, index_col=0, sep='\t')
                df1 = pd.merge(df1,df2,left_index=True,right_index=True)
                d = df1[df1.SADLN == 1]
                plt.scatter(d.values[:,0], d.values[:,1], c='r', label='c1')

                d = df1[df1.SADLN == 2]
                plt.scatter(d.values[:,0], d.values[:,1], c='y', label='c2')

                d = df1[df1.SADLN == 3]
                plt.scatter(d.values[:,0], d.values[:,1], c='g', label='c3')

                d = df1[df1.SADLN == 4]
                plt.scatter(d.values[:,0], d.values[:,1], c='b', label='c4')

                d = df1[df1.SADLN == 5]
                plt.scatter(d.values[:,0], d.values[:,1], c='c', label='c5')
                plt.xlabel("tSNE Dimension1", fontsize=14)
                plt.ylabel("tSNE Dimension2", fontsize=14)
                plt.legend(['c1', 'c2','c3','c4'],title="Cluster",loc=1, bbox_to_anchor=(1.105,1.0),borderaxespad = 0.)
                #plt.legend(['c1', 'c2', 'c3'], title="Cluster")
                plt.show()
        # cancer_type = args.type
        # fea_tmp_file = './fea/' + cancer_type + '.vae'
        # out_file = './fea/' + cancer_type + '.tsne'
        # label_file = './results/' + cancer_type + '.vae'
        # if isfile(fea_tmp_file):
        #     df1 = pd.read_csv(fea_tmp_file, header=0, index_col=0, sep='\t')
        #     mat = df1.values.astype(float)
        #     labels = VAE.tsne(mat)
        #     print(labels.shape)
        #     df1['x'] = labels[:, 0]
        #     df1['y'] = labels[:, 1]
        #     df1 = df1.loc[:, ['x', 'y']]
        #     df1.to_csv(out_file, header=True, index=True, sep='\t')
        #     if isfile(label_file):
        #         df2 = pd.read_csv(label_file, header=0, index_col=0, sep='\t')
        #         df1 = pd.merge(df1, df2, left_index=True, right_index=True)
        #         d = df1[df1.vae == 1]
        #         plt.scatter(d.values[:, 0], d.values[:, 1], c='r', label='c1')
        #
        #         d = df1[df1.vae == 2]
        #         plt.scatter(d.values[:, 0], d.values[:, 1], c='y', label='c2')
        #
        #         d = df1[df1.vae == 3]
        #         plt.scatter(d.values[:, 0], d.values[:, 1], c='g', label='c3')
        #
        #         d = df1[df1.vae == 4]
        #         plt.scatter(d.values[:, 0], d.values[:, 1], c='b', label='c4')
        #
        #         d = df1[df1.vae == 5]
        #         plt.scatter(d.values[:, 0], d.values[:, 1], c='c', label='c5')
        #         plt.xlabel("tSNE Dimension1", fontsize=14)
        #         plt.ylabel("tSNE Dimension2", fontsize=14)
        #         plt.legend(['c1', 'c2', 'c3', 'c4','c5'], title="Cluster", loc=1, bbox_to_anchor=(1.105, 1.0),
        #                            borderaxespad=0.)
        #         plt.show()
        else:
            print('file does not exist!')

    elif args.run_mode == 'kmeans':
        cancer_type = args.type
        if cancer_type not in cancer_dict and args.cluster_num == -1:
            print("Please set the number of clusters!")
        elif args.cluster_num == -1:
            args.cluster_num = cancer_dict[cancer_type]
        dfs = []
        for line in open(args.file_input, 'rt'):
            base_file = splitext(basename(line.rstrip()))[0]
            fea_tmp_file = './fea/' + cancer_type + '/' + base_file + '.fea'
            dfs.append(pd.read_csv(fea_tmp_file, header=0, index_col=0, sep=','))
        X = pd.concat(dfs, axis=0).T
        print(X.head(5))
        print(X.shape)
        X['kmeans'] = SADLN.kmeans(args.cluster_num).fit_predict(X.values) + 1
        X = X.loc[:, ['kmeans']]
        out_file = './results/' + cancer_type + '.kmeans'
        X.to_csv(out_file, header=True, index=True, sep='\t')

    elif args.run_mode == 'spectral':
        cancer_type = args.type
        if cancer_type not in cancer_dict and args.cluster_num == -1:
            print("Please set the number of clusters!")
        elif args.cluster_num == -1:
            args.cluster_num = cancer_dict[cancer_type]
        dfs = []
        for line in open(args.file_input, 'rt'):
            base_file = splitext(basename(line.rstrip()))[0]
            fea_tmp_file = './fea/' + cancer_type + '/' + base_file + '.fea'
            dfs.append(pd.read_csv(fea_tmp_file, header=0, index_col=0, sep=','))
        X = pd.concat(dfs, axis=0).T
        print(X.head(5))
        print(X.shape)
        X['spectral'] = SADLN.spectral(args.cluster_num).fit_predict(X.values) + 1
        X = X.loc[:, ['spectral']]
        out_file = './results/' + cancer_type + '.spectral'
        X.to_csv(out_file, header=True, index=True, sep='\t')

    elif args.run_mode == 'ae':
        cancer_type = args.type
        if cancer_type not in cancer_dict and args.cluster_num == -1:
            print("Please set the number of clusters!")
        elif args.cluster_num == -1:
            args.cluster_num = cancer_dict[cancer_type]
        dfs = []
        for line in open(args.file_input, 'rt'):# 遍历读取四种类型的数据到dfs变量中
            base_file = splitext(basename(line.rstrip()))[0]
            fea_tmp_file = './fea/' + cancer_type + '/' + base_file + '.fea'
            dfs.append(pd.read_csv(fea_tmp_file, header=0, index_col=0, sep=','))
        X = pd.concat(dfs, axis=0).T
        print(X.head(5))
        print(X.shape)
        fea_save_file = './fea/' + cancer_type + '.ae'
        start_time = time.time()
        vec = SADLN.feature_ae(X, n_components=100)  # 使用feature_ae提取latent code特征
        df = pd.DataFrame(data=[time.time() - start_time])
        vec.to_csv(fea_save_file, header=True, index=True, sep='\t') # 保存latent coce到fea_save_file文件中
        out_file = './results/' + cancer_type + '.ae.time'
        df.to_csv(out_file, header=True, index=False, sep=',')
        # 已将数据通过gan转为latent code
        ###############################################################################################################
        # 基于latent code做了gmm聚类操作
        if isfile(fea_save_file):
            X = pd.read_csv(fea_save_file, header=0, index_col=0, sep='\t')
            X['ae'] = SADLN.gmm(args.cluster_num).fit_predict(X.values) + 1
            X = X.loc[:, ['ae']]
            out_file = './results/' + cancer_type + '.ae'
            X.to_csv(out_file, header=True, index=True, sep='\t')
        else:
            print('--------------file does not exist!')

    elif args.run_mode == 'vae':
        cancer_type = args.type
        if cancer_type not in cancer_dict and args.cluster_num == -1:
            print("Please set the number of clusters!")
        elif args.cluster_num == -1:
            args.cluster_num = cancer_dict[cancer_type]
        dfs = []
        for line in open(args.file_input, 'rt'):
            base_file = splitext(basename(line.rstrip()))[0]
            fea_tmp_file = './fea/' + cancer_type + '/' + base_file + '.fea'
            dfs.append(pd.read_csv(fea_tmp_file, header=0, index_col=0, sep=','))
        X = pd.concat(dfs, axis=0).T # 将四组模态数据合成一组数据
        print(X.head(5))
        print(X.shape)
        fea_save_file = './fea/' + cancer_type + '.vae'
        start_time = time.time()
        vec = SADLN.feature_vae(X, n_components=100)
        df = pd.DataFrame(data=[time.time() - start_time])
        vec.to_csv(fea_save_file, header=True, index=True, sep='\t')
        out_file = './results/' + cancer_type + '.vae.time'
        df.to_csv(out_file, header=True, index=False, sep=',')
        if isfile(fea_save_file):
            X = pd.read_csv(fea_save_file, header=0, index_col=0, sep='\t')
            X['vae'] = SADLN.gmm(args.cluster_num).fit_predict(X.values) + 1
            X = X.loc[:, ['vae']]
            out_file = './results/' + cancer_type + '.vae'
            X.to_csv(out_file, header=True, index=True, sep='\t')
        else:
            print('file does not exist!')

    elif args.run_mode == 'cc':
        K1_dict = {'BRCA': 4, 'BLCA': 3, 'KIRC': 3,
                   'GBM': 2, 'LUAD': 3, 'PAAD': 2,
                   'SKCM': 3, 'STAD': 3, 'UCEC': 4, 'UVM': 2}
        K2_dict = {'BRCA': 8, 'BLCA': 6, 'KIRC': 6,
                   'GBM': 4, 'LUAD': 6, 'PAAD': 4,
                   'SKCM': 6, 'STAD': 6, 'UCEC': 8, 'UVM': 4}
        cancer_type = args.type
        base_file = splitext(basename(args.file_input))[0]
        fea_tmp_file = './fea/' + cancer_type + '.fea'
        fs = []
        cc_file = './results/k.cc'
        fp = open(cc_file, 'a')
        if isfile(fea_tmp_file):
            X = pd.read_csv(fea_tmp_file, header=0, index_col=0, sep='\t')
            cc = ConsensusCluster(SADLN.gmm, K1_dict[cancer_type], K2_dict[cancer_type], 10)
            cc.fit(X.values)
            X['cc'] = SADLN.gmm(cc.bestK).fit_predict(X.values) + 1
            X = X.loc[:, ['cc']]
            out_file = './results/' + cancer_type + '.cc'
            X.to_csv(out_file, header=True, index=True, sep='\t')
            fp.write("%s, k=%d\n" % (cancer_type, cc.bestK))
        else:
            print('file does not exist!')
        fp.close()


if __name__ == "__main__":
    main()
