class Config:
    def __init__(self, data_type, dataset_index, train_ratio, hidden_units,
                 epochs, batch_size, normalization, save_best_model, early_stopping, sample_weight,
                 verbose, additional_prediction, seed):
        self.data_type = data_type
        self.dataset_index = dataset_index
        self.train_ratio = train_ratio
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.normalization = normalization
        self.save_best_model = save_best_model
        self.early_stopping = early_stopping
        # self.patience = patience  # 早期終了のパラメータが必要な場合、引数に追加
        self.sample_weight = sample_weight
        self.verbose = verbose
        self.additional_prediction = additional_prediction
        self.seed = seed
