# LibriSpeech Phoneme Classification

Kaggle 竞赛地址：[Frame-wise phoneme classification on the LibriSpeech Dataset.](https://www.kaggle.com/c/ml2022spring-hw2)

## 数据与任务描述

在任务描述的 PPT 中（./resources/description_slides.ppt），涉及了一些 Speech 背景的知识，难免容易将缺乏 Speech 背景的同学吓到，但实际上，不了解这些知识完全没有关系。只需要记住，这是一个 41 分类的多分类任务就好了。

接下来介绍一下数据的格式。数据文件的目录树如图：

![数据文件目录树](./imgs/dir_tree.png)

其中，_feat_ 目录包含了训练和测试的数据文件，以训练数据为例：

![训练数据](./imgs/train_pt_data.png)

每一份 `.pt` 文件都对应一条语音，一条语音可以通过 `torch.load` 加载得到一个形状为 `(T, 39)` 的 tensor 数据，其中 T 代表时间维度，39 代表特征向量的特征维度（MFCC）。时间维度 `T` 上的每一个 39 维的特征向量，都分别代表了一个小小的时间窗口（frame）所对应的音频的特征表示。任务需要做的就是预测每个 frame 中的音频发音属于哪一个 phoneme。至于 phoneme，可以理解成类似音标的东西，一共有 41 类 phoneme，所以这是一个 41 分类的任务。

_test_split.txt_、_train_split.txt_ 和 _train_labels.txt_ 都是元数据文件。其中 _train_split.txt_ 和 _test_split.txt_ 文件中保存了 _feat_ 目录下的数据文件的文件名，一行对应一个文件：

![train split txt](./imgs/train_split_txt.png)

_train_labels.txt_ 指明了训练数据中每个 frame 的 MFCC 特征与 phoneme 标签的关系：

![train_labels_txt](./imgs/train_labels_txt.png)

如图，每一行代表一个文件的信息，每个文件的信息通过 `' '` 分隔，每一行首先是文件名，后面跟的数字代表了该文件加载后，每个 frame 的类别。我们没有 _test_labels.txt_，我们的任务就是用训练好的模型，然后预测 _test_split.txt_ 中的文件的每个 frame 所属的 label。

## 解法与思路

我主要尝试了 MLP 和 RNN 两种架构，其中 MLP 架构可以很轻松的通过 Medium baseline，如果非常精细的调参可以过 Strong baseline；RNN 类型的架构可以很轻松的通过 Strong baseline，通过调参可以过 Boss baseline。

### Feature Scaling

无论哪种架构，对特征做 feature scaling 都非常有帮助，可以加速模型的优化速度，以 MLP 为例，不使用 feature scaling 的情况下，用了五层 1024 的 MLP 的效果，和单层 512 + 特征标准化后的效果差不多。如果不做 feature scaling，单层 512 的 MLP 远远差于标准化后同样架构的 MLP。

我在该 repo 中使用了 sklearn 库对特征进行了 z-score scaling，效果提升非常显著，sklearn 还支持其他的 scaling 方法，感兴趣的话可以尝试。

### MLP 架构

如果使用 MLP 架构，在预测一个 frame 的类别时，需要拼接左右两边相邻的 frame 共同作为分类特征，一般来说拼接的相邻 frame 越多，模型的分类性能越好，当然也不要过多，容易过拟合。我在实验中选择了左右各拼接 7 个 frame，算上要预测的 frame，共 15 个 frame。对于边界的 frame，拼接时会遇到左边/右边的 frame 不够的情况，需要 padding，实验中使用了 zero padding 和镜像 padding，具体可见代码。

MLP 架构通过调整隐藏层的宽度、层数（深度）、batch normalization 和 dropout，比较容易的通过 Medium baseline。如果想要通过 Strong baseline，需要很精细的调参，还要考虑优化器的参数，以及增加训练的步数，在实验中发现 batch normalization 对模型性能有显著提升。

### RNN 架构

本次 Kaggle 属于 sequence labeling 任务，使用时序模型更符合直觉（在 PPT 中，宏毅老师也有提示想过 boss baseline 要用 RNN 架构）。如果你不想尝试 MLP 架构的调参实验，可以直接使用 RNN 架构，本 repo 使用了 PyTorch 中的 `PackedSequence` 完成序列数据的处理，并尝试了 simple RNN、LSTM 和 GRU 三种架构，对每一个时间步的输出和其对应的 phoneme 类别做比较，使用交叉熵损失函数进行反向传播，属于 many-to-many 任务，具体详见代码。

在该任务中，双向 rnn 的表现远超单向 rnn，这点也非常符合直觉。整体来说，使用 RNN 架构，即使是最简单的双向 simple RNN，也可以比较轻松的通过 Strong baseline。如果使用 Bi-LSTM 或者 Bi-GRU，性能表现会远超 simple RNN，而 LSTM 和 GUR 这两种架构的表现基本一致，随便选一个就可以。对 Bi-LSTM 和 Bi-GRU 进行精细调参，可以超过 Boss baseline。
