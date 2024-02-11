import numpy as np
from scipy.stats import truncnorm
from sklearn.preprocessing import MinMaxScaler

import srdata


class Data:
    def __init__(self, nn_failure_prediction):
        self.nn_failure_prediction = nn_failure_prediction

    def make_dataset(self, size=100, scale=50, shape=2, loc=0):
        # シミュレーション
        if self.nn_failure_prediction.config.data_type == 1:
            # シードの固定
            np.random.seed(self.nn_failure_prediction.config.seed)

            # 指数分布モデル
            if self.nn_failure_prediction.config.dataset_index == 1:
                # データの生成
                samples = np.random.exponential(scale, size)
                sorted_samples = np.sort(samples)

                # データの保存
                with open("datasets/simulaton/exponential.csv", "w") as f:
                    for sample in sorted_samples:
                        f.write(str(sample) + "\n")

            # ガンマ分布モデル
            # 正式な分布と違う？(https://github.com/SwReliab/SRATS2017/blob/master/docs/pdfs/srats_model.pdf)
            # こっちがあっているかも(https://www.researchgate.net/profile/Hiroyuki-Okamura/publication/261238504_SRATS_Software_reliability_assessment_tool_on_spreadsheet_Experience_report/links/5788308e08ae21394a0c7b1e/SRATS-Software-reliability-assessment-tool-on-spreadsheet-Experience-report.pdf)
            # 将来的に逆変換サンプリングを用いて実装する予定
            elif self.nn_failure_prediction.config.dataset_index == 2:
                # データの生成
                samples = np.random.gamma(shape, scale, size)
                sorted_samples = np.sort(samples)

                # データの保存
                with open("datasets/simulaton/gamma.csv", "w") as f:
                    for sample in sorted_samples:
                        f.write(str(sample) + "\n")

            # 切断正規分布モデル
            # 正式な分布と違う？(https://github.com/SwReliab/SRATS2017/blob/master/docs/pdfs/srats_model.pdf)
            elif self.nn_failure_prediction.config.dataset_index == 3:
                # データの生成
                truncnorm_dist = truncnorm(
                    (0 - loc) / scale, (np.inf - 0) / scale, loc=loc, scale=scale)
                samples = truncnorm_dist.rvs(size)
                sorted_samples = np.sort(samples)

                # データの保存
                with open("datasets/simulaton/truncnorm.csv", "w") as f:
                    for sample in sorted_samples:
                        f.write(str(sample) + "\n")

            # データの作成
            t = sorted_samples.reshape(-1, 1)
            y = np.arange(1, len(sorted_samples) + 1).reshape(-1, 1)

        # 実データ
        elif self.nn_failure_prediction.config.data_type == 2:
            dataset_names = ["Lyu/J1.csv", "Lyu/J2.csv",
                             "Lyu/J3.csv", "Lyu/J4.csv", "Lyu/J5.csv"]
            data_df = srdata.get_dataset(
                dataset_names[self.nn_failure_prediction.config.dataset_index - 1])
            data = data_df.iloc[:, 0].values
            cum_data = data.cumsum()
            t = np.arange(1, len(cum_data) + 1).reshape(-1, 1)
            y = cum_data.reshape(-1, 1)

        # メンバ変数として保存
        self.nn_failure_prediction.result.t = t
        self.nn_failure_prediction.result.y = y

    def normalize(self):
        self.nn_failure_prediction.result.scaler_t = MinMaxScaler(
            feature_range=(0, 1))
        self.nn_failure_prediction.result.scaler_y = MinMaxScaler(
            feature_range=(0, 1))
        self.nn_failure_prediction.result.t = self.nn_failure_prediction.result.scaler_t.fit_transform(
            self.nn_failure_prediction.result.t)
        self.nn_failure_prediction.result.y = self.nn_failure_prediction.result.scaler_y.fit_transform(
            self.nn_failure_prediction.result.y)

    # def denormalize(self, t_normalized, y_normalized):
    #     t_denormalized = self.nn_failure_prediction.result.scaler_t.inverse_transform(t_normalized)
    #     y_denormalized = self.nn_failure_prediction.result.scaler_y.inverse_transform(y_normalized)

    #     return t_denormalized, y_denormalized

    def split_dataset(self):
        # データの分割
        train_size = int(len(self.nn_failure_prediction.result.t) *
                         self.nn_failure_prediction.config.train_ratio)  # 小数点以下切り捨て

        # データの分割
        self.nn_failure_prediction.result.t_train = self.nn_failure_prediction.result.t[
            :train_size]
        self.nn_failure_prediction.result.y_train = self.nn_failure_prediction.result.y[
            :train_size]
        self.nn_failure_prediction.result.t_test = self.nn_failure_prediction.result.t[
            train_size:]
        self.nn_failure_prediction.result.y_test = self.nn_failure_prediction.result.y[
            train_size:]

    def pipeline(self):
        self.make_dataset()
        if self.nn_failure_prediction.config.normalization:
            self.normalize()
        self.split_dataset()
