from offline.data_preprocess import DataPreprocess
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models.deepfm import FM
import tensorflow.keras as keras

user_num = 4691
movie_num = 2514
year_num = 76
genre_num = 9

embedding_size = 16
epoch = 5
batch_size = 512

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

model = FM(linear_feature_columns, dnn_feature_columns, task='binary')
model.summary()
model.compile(optimizer="adam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

model.fit(train_dataset, epochs=epoch, steps_per_epoch=1270010 // batch_size + 1,
          validation_data=test_dataset, validation_steps=159256 // batch_size + 1,
          )
