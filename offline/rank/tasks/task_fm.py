from offline.data_preprocess import DataPreprocess
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
import numpy as np
import os
from offline.rank.models.metrics import get_callback
from offline.rank.models.layer import FMLayer
from offline.rank.models.fm import FM
from offline.rank.inputs import get_input_tensor_data
from offline.rank.models.metrics import get_metrics
from util.config import *
"""

FM 这里最优的模型是：Embedding_size = 16,添加sigmoid 
acc:  0.89040219  f1  0.7039

当然跟 DeepFM：f1:0.74 及 Din:f1:0.75 还是有很大的差距

FM 模型

embedding_size = 16
模型预测最终指标
参数量大约：123,932
测试集：acc:  0.88480738 precision:  0.7634391823359601 recall:  0.6144234089981476 f1:  0.680873271288162

embedding_size = 32
模型预测最终指标
参数量大约：240,570
测试集：acc:  0.88715582 precision:  0.7753749702404571 recall:  0.6135129195315688 f1:  0.6850121816556535


在FM 最后一层输出层，添加了sigmoid 模型得到了不小的提升
embedding_size = 16
模型预测最终指标
参数量大约：123,932
测试集：acc:  0.89040219 precision:  0.7655195308177493 recall:  0.6515964961853631 f1:  0.7039788338251756


在FM 最后一层输出层，添加了sigmoid 模型并没有得到提升，反而下降厉害，不能收敛。
embedding_size = 32
模型预测最终指标
参数量大约：240,570
测试集：acc:  0.838228 precision:  0.6191669276542436 recall:  0.4965621173589526 f1:  0.5511281470511368

总结：

针对FM，Embedding_size 设置重要，是否 添加sigmoid 也很重要，需要不停的尝试。




"""

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.compat.v1.disable_eager_execution()


class TaskFM(object):
    user_num = 4691
    movie_num = 2514
    year_num = 76
    genre_num = 9

    embedding_size = 16

    batch_size = 512

    version = "0001"

    epoch = 20

    model_path = PROJECT_PATH + 'offline/rank/tasks/checkpoint/fm_model/'

    def __init__(self, version="0001"):
        self.data_process = DataPreprocess()

        self.version = version

        model_path = self.model_path + self.version

        if os.path.exists(model_path):
            self.fm_model = tf.saved_model.load(model_path)

    def train(self):
        self.train_dataset, \
        self.test_dataset = self.data_process.generate_data(batch_size=self.batch_size,
                                                            epoch=self.epoch)

        fm = FM(user_num=self.user_num,
                movie_num=self.movie_num,
                year_num=self.year_num,
                genre_num=self.genre_num,
                embedding_size=self.embedding_size,
                is_sigmoid=True)

        fm_model = fm.get_fm_model()
        fm_model.summary()
        fm_model.compile(optimizer="adam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

        fm_model.fit(self.train_dataset, epochs=self.epoch, steps_per_epoch=1270010 // self.batch_size + 1,
                     callbacks=[get_callback(self.test_dataset, self.model_path, self.version)]
                     )

    def predict(self, test_dataset=None, is_print=True):

        if test_dataset is None:
            test_dataset = self.data_process.generate_test_data()

        true_y = []
        pred_y = []
        for test_x, test_y in test_dataset:
            data = get_input_tensor_data(test_x)
            array = self.fm_model(data, training=False)
            _array = np.where(array > 0.5, 1, 0)
            pred_y.extend(_array)
            true_y.extend(test_y.numpy()[:, np.newaxis])

        get_metrics(true_y, pred_y, is_print=is_print)


if __name__ == '__main__':
    import time

    time_time = time.time()
    TaskFM().train()
    print(time.time() - time_time)
