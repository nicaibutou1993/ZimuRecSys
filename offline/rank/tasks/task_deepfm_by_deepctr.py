from offline.data_preprocess import DataPreprocess
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models.deepfm import DeepFM
import tensorflow.keras as keras
from offline.rank.models.metrics import get_callback


"""
deepfm:
embedding_size = 16
参数量：155,035
测试集：acc:  [0.90004709] precision:  0.7606071510353626 recall:  0.7299927788766444 f1:  0.7449855815443769

2481/2481 [==============================] - 67s 27ms/step - loss: 0.2932 - accuracy: 0.8748
Epoch 2/20
2478/2481 [============================>.] - ETA: 0s - loss: 0.2185 - accuracy: 0.9110acc:  [0.89730935] precision:  0.7546754313886607 recall:  0.7208878842108568 f1:  0.7373948230457963
2481/2481 [==============================] - 62s 25ms/step - loss: 0.2185 - accuracy: 0.9110
Epoch 3/20
2474/2481 [============================>.] - ETA: 0s - loss: 0.1969 - accuracy: 0.9216acc:  [0.89902986] precision:  0.7718106924959498 recall:  0.7029920567643088 f1:  0.7357957346127303
2481/2481 [==============================] - 59s 24ms/step - loss: 0.1969 - accuracy: 0.9216
Epoch 4/20
2480/2481 [============================>.] - ETA: 0s - loss: 0.1848 - accuracy: 0.9277acc:  [0.90004709] precision:  0.7606071510353626 recall:  0.7299927788766444 f1:  0.7449855815443769
2481/2481 [==============================] - 62s 25ms/step - loss: 0.1847 - accuracy: 0.9277
Epoch 5/20
2479/2481 [============================>.] - ETA: 0s - loss: 0.1769 - accuracy: 0.9317acc:  [0.89955731] precision:  0.7690297621067635 recall:  0.711469027660042 f1:  0.7391304347826088
2481/2481 [==============================] - 59s 24ms/step - loss: 0.1769 - accuracy: 0.9317
Epoch 6/20
2476/2481 [============================>.] - ETA: 0s - loss: 0.1711 - accuracy: 0.9347acc:  [0.89904242] precision:  0.7644612856711713 recall:  0.7157389092964114 f1:  0.7392982228564016
2481/2481 [==============================] - 59s 24ms/step - loss: 0.1711 - accuracy: 0.9347
Epoch 7/20
2476/2481 [============================>.] - ETA: 0s - loss: 0.1667 - accuracy: 0.9371acc:  [0.89751028] precision:  0.7929226242124722 recall:  0.6598850899500801 f1:  0.7203125535487852
2481/2481 [==============================] - 60s 24ms/step - loss: 0.1667 - accuracy: 0.9371
Epoch 8/20
2478/2481 [============================>.] - ETA: 0s - loss: 0.1630 - accuracy: 0.9390acc:  [0.90009733] precision:  0.7677449695992475 recall:  0.717559888229569 f1:  0.7418046088932165
2481/2481 [==============================] - 60s 24ms/step - loss: 0.1630 - accuracy: 0.9389
Epoch 9/20
2477/2481 [============================>.] - ETA: 0s - loss: 0.1601 - accuracy: 0.9405acc:  [0.89896707] precision:  0.7641006736150675 recall:  0.7158330978619195 f1:  0.739179769816826
2481/2481 [==============================] - 60s 24ms/step - loss: 0.1601 - accuracy: 0.9405




"""

user_num = 4691
movie_num = 2514
year_num = 76
genre_num = 9

embedding_size = 16
epoch = 20
batch_size = 512

model_path = "checkpoint/deepfm_model_by_deepctr/"

data_process = DataPreprocess()
train_dataset, test_dataset = data_process.generate_data(batch_size=batch_size, epoch=epoch)

feature_columns = [SparseFeat("user_id", vocabulary_size=user_num, embedding_dim=embedding_size),
                   SparseFeat("movie_id", vocabulary_size=movie_num, embedding_dim=embedding_size),
                   SparseFeat("current_label", vocabulary_size=genre_num, embedding_dim=embedding_size),
                   SparseFeat("release_year", vocabulary_size=year_num, embedding_dim=embedding_size),
                   ]

feature_columns += [VarLenSparseFeat(
    SparseFeat("user_recent_click_movie_ids", vocabulary_size=movie_num, embedding_dim=embedding_size,
               embedding_name='movie_id'), maxlen=20),
    VarLenSparseFeat(
        SparseFeat("user_recent_click_labels", vocabulary_size=genre_num, embedding_dim=embedding_size,
                   embedding_name='current_label'), maxlen=20),
    VarLenSparseFeat(
        SparseFeat("user_like_genres", vocabulary_size=genre_num, embedding_dim=embedding_size,
                   embedding_name='current_label'), maxlen=2),
]

dnn_feature_columns = feature_columns
linear_feature_columns = feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
model.summary()
model.compile(optimizer="adam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

model.fit(train_dataset, epochs=epoch,
          steps_per_epoch=1270010 // batch_size + 1,
          callbacks=[get_callback(test_dataset, model_path)])
