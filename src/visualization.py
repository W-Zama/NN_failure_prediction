import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, nn_failure_prediction):
        self.nn_failure_prediction = nn_failure_prediction

    def visualize_loss(self):
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, self.nn_failure_prediction.config.epochs + 1),
                 self.nn_failure_prediction.result.history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        if self.nn_failure_prediction.config.normalization:
            plt.ylim(0, 0.01)   # 正規化する場合ylimを制限しないと損失関数が小さくなりすぎて見えなくなる
        plt.legend(['Train'], loc='upper left')
        plt.show()

    def visualize_prediction(self):
        # 予測値のプロット
        plt.figure(figsize=(12, 6))
        plt.plot(self.nn_failure_prediction.result.t_train, self.nn_failure_prediction.result.y_train,
                 color="red", label='True_train', marker='o', markersize=2)
        plt.plot(self.nn_failure_prediction.result.t_test, self.nn_failure_prediction.result.y_test,
                 color="blue", label='True_test', marker='o', markersize=2)
        plt.plot(self.nn_failure_prediction.result.pred_t, self.nn_failure_prediction.result.predictions,
                 color="green", label='Predicted', marker='o', markersize=2)
        plt.title('Prediction')
        plt.xlabel('Testing Time')
        plt.ylabel('Cumulative Number of Failures')
        plt.legend()
        plt.show()
