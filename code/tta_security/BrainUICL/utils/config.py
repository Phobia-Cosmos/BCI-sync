ISRUC_DICT = {
    "pretrain_lr": 1e-4,
    "ssl_lr": 1e-6,
    "incremental_lr": 1e-7,
    "batch": 16
}


class ModelConfig(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.ConvDrop = 0.1
        self.EncoderParam = EncoderConfig()
        self.SleepMlpParam = SleepMlpParam()
        self.FaceMlpParam = FaceMlpParam()
        self.BCI2000MlpParam = BCI2000MlpParam()
        self.NumClasses = 5
        self.ClassNames = ['W', 'N1', 'N2', 'N3', 'REM']
        self.ClassNamesFace = ['Anger',
                               'Disgust',
                               'Fear',
                               'Sadness',
                               'Neutral',
                               'Amusement',
                               'Inspiration',
                               'Joy',
                               'Tenderness']
        self.ClassNamesBCI2000 = ['Left', 'Right', 'Fist', 'Feet']
        self.SeqLength = 20
        # TODO:为什么BatchSize是32？
        # 论文 Table 8 将 32 作为三项任务共享的 sequence batch；实际 runner
        # 仍以命令行 --batch 为准，公开仓库的命令行默认值 16 与论文表不一致。
        self.BatchSize = 32
        self.EpochLength = 3000
        self.EpochLengthFace = 7500
        self.EpochLengthBCI2000 = 640
        # TODO:那如果对于其他数据集怎么处理呢？
        # 数据集分支只设置输入通道、每个 epoch 的采样点数和输出类别数；
        # CNN/Transformer/MLP 主体超参数按论文保持共享。FACED 的完整公开
        # frontend 不存在，因此当前 1/31 接口属于本地可运行重建。
        channel_num = self.get_channel_info()
        if self.dataset == "ISRUC":
            self.EegNum = channel_num[0]
            self.EogNum = channel_num[1]
        elif self.dataset == "FACED":
            self.NumClasses = 9
            self.ClassNames = list(self.ClassNamesFace)
            self.EpochLength = 2500
            # Keep the common (auxiliary, EEG) runner interface. The FACED
            # frontend rejoins these tensors before its 32-channel convolution.
            self.EegNum = channel_num[0]
            self.EogNum = channel_num[1]

    def get_channel_info(self):
        if self.dataset == "ISRUC":
            return [6, 2]
        if self.dataset == "FACED":
            return [31, 1]

# TODO:各个参数为什么取这些值？随机的还是怎么弄出来的？为什么MLP的结构除了最后的输出不同其他都是一样的？MLP的参数又是如何生成的？
# 8 heads、512 hidden、3 layers、dropout 0.1 和共享 MLP hidden sizes 来自
# 论文统一结构设置，不是运行时随机挑选；只有最后一层维度随类别数改变。
# 各层权重仍由随机初始化后通过 source-supervised pretraining 学得并保存，
# 这里的列表只声明网络维度，不生成训练后的参数值。
class EncoderConfig(object):
    def __init__(self):
        self.n_head = 8
        self.d_model = 512
        self.layer_num = 3
        self.drop = 0.1


class SleepMlpParam(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [512, 256]
        self.second_linear = [256, 128]
        self.out_linear = [128, 5]


class FaceMlpParam(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [512, 256]
        self.second_linear = [256, 128]
        self.out_linear = [128, 9]


class BCI2000MlpParam(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [512, 256]
        self.second_linear = [256, 128]
        self.out_linear = [128, 4]
