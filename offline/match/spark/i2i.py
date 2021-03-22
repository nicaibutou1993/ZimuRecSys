from util.config import PROJECT_PATH
import pandas as pd
from scipy.spatial import distance
import numpy as np
import json

pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)

path = PROJECT_PATH + 'data/movies.dat'
movie_embedding = pd.read_csv('i2i_movie_embedding.csv', index_col=0)


movie_embedding.rename(columns={'word': 'movie_id'}, inplace=True)

movie_df = pd.read_csv(path, sep="::", header=None)
movie_df.columns = ["movie_id", 'name', 'genres']
movie_df = movie_df[['movie_id', 'genres']]
print(movie_df)
movie_df = pd.merge(movie_embedding, movie_df, how='inner', on='movie_id')

movie_df['vector'] = movie_df['vector'].map(lambda x: np.array(json.loads(x)))

'''指定某一个电影，查找相似的电影'''
movie_id = 383

current_movie = movie_df.loc[movie_df['movie_id'] == movie_id, 'vector'].iloc[0]

"""cosine 相似度计算"""
movie_df["sim_value"] = movie_df['vector'].apply(lambda x: 1 - distance.cosine(current_movie, x))

movie_df = movie_df.sort_values('sim_value', ascending=False).head(20)
print(movie_df)
