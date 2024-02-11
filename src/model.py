import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from keras.callbacks import ModelCheckpoint


class PredictiveModel:
    def __init__(self, nn_failure_prediction):
        self.nn_failure_prediction = nn_failure_prediction
        tf.keras.utils.set_random_seed(self.nn_failure_prediction.config.seed)

    def define_model(self):
        input_layer = Input(shape=(1,))
        x = Dense(self.nn_failure_prediction.config.hidden_units,
                  activation="relu")(input_layer)
        x = Dense(self.nn_failure_prediction.config.hidden_units,
                  activation="relu")(x)
        x = Dense(self.nn_failure_prediction.config.hidden_units,
                  activation="relu")(x)
        x = Dense(1, activation="sigmoid")(x)
        output_layer = Dense(
            1, use_bias=False, kernel_initializer=Constant(value=1.0))(x)

        self.nn_failure_prediction.result.model = Model(
            inputs=input_layer, outputs=output_layer)

    def compile_model(self):
        self.nn_failure_prediction.result.model.compile(
            loss='mse', optimizer='adam')

    def train_model(self):
        # ModelCheckpointの設定
        callbacks_list = []
        if self.nn_failure_prediction.config.save_best_model:
            checkpoint = ModelCheckpoint(
                './models/best_model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list.append(checkpoint)

        # サンプルごとの重み
        sample_weight_array = None
        if self.nn_failure_prediction.config.sample_weight:
            sample_weight_array = np.array(
                [i for i in range(1, len(self.nn_failure_prediction.result.t_train) + 1)])

        # 学習
        self.nn_failure_prediction.result.history = self.nn_failure_prediction.result.model.fit(self.nn_failure_prediction.result.t_train, self.nn_failure_prediction.result.y_train, epochs=self.nn_failure_prediction.config.epochs, batch_size=self.nn_failure_prediction.config.batch_size, verbose=self.nn_failure_prediction.config.verbose, sample_weight=sample_weight_array, callbacks=callbacks_list)

        if (self.nn_failure_prediction.config.save_best_model):
            self.nn_failure_prediction.result.model = load_model('./models/best_model.h5')
            min_loss_index = np.argmin(
                self.nn_failure_prediction.result.history.history['loss'])
            print("Best model is from epoch {} and loss is {}".format(
                min_loss_index + 1, self.nn_failure_prediction.result.history.history['loss'][min_loss_index]))

    def predict_model(self):
        self.nn_failure_prediction.result.pred_t = self.nn_failure_prediction.result.t
        if self.nn_failure_prediction.config.additional_prediction and self.nn_failure_prediction.config.normalization:
            self.nn_failure_prediction.result.pred_t = np.append(
                self.nn_failure_prediction.result.pred_t, np.linspace(1, 2, 10).reshape(-1, 1))
        self.nn_failure_prediction.result.predictions = self.nn_failure_prediction.result.model.predict(
            self.nn_failure_prediction.result.pred_t)

    def save_model(self):
        pass
        # h5形式で保存
        # model.save_weights('../../models/weights/weights.h5')

        # パラメータをテキスト形式で保存
        # weights = model.get_weights()
        # with open('../../models/weights/model_weights.txt', 'w') as f:
        #     for layer_weights in weights:
        #         np.savetxt(f, layer_weights, fmt='%s')

    def pipeline(self):
        self.define_model()
        self.compile_model()
        self.train_model()
        self.predict_model()
        self.save_model()
