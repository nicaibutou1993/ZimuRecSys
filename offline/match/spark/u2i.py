import pandas as pd
from util.config import PROJECT_PATH
import numpy as np
import json
from scipy.spatial import distance

pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)

user_embedding_df = pd.read_csv('u2i_user_embedding.csv',index_col=0)
movie_embedding_df = pd.read_csv('u2i_movie_embedding.csv',index_col=0)

movie_embedding_df.rename(columns={'id':'movie_id'},inplace=True)

movie_embedding_df["features"] = movie_embedding_df["features"].map(lambda x : np.array(json.loads(x)))
user_embedding_df["features"] = user_embedding_df["features"].map(lambda x : np.array(json.loads(x)))

path = PROJECT_PATH + 'data/movies.dat'
movie_df = pd.read_csv(path, sep="::", header=None)
movie_df.columns = ["movie_id", 'name', 'genres']
movie_df = movie_df[['movie_id', 'genres']]


rating_path = PROJECT_PATH + 'data/ratings.dat'
rating_df = pd.read_csv(rating_path, sep="::", header=None)
rating_df.columns = ["user_id", 'movie_id', 'score','time']


movie_embedding_df = pd.merge(movie_embedding_df,movie_df,on='movie_id')

print(user_embedding_df.head(3))
print(movie_embedding_df.head(3))

user_id = 1

current_track_df = rating_df[rating_df['user_id'] == user_id]

#print(rating_df[rating_df['movie_id'].isin([3437,1543,1864])])
#print(rating_df[rating_df['movie_id'].isin([2858,260,1196])])

#print(rating_df.groupby('movie_id').count().sort_values('user_id',ascending=False))
print(current_track_df)


user_embedding = user_embedding_df.loc[user_embedding_df['id'] == user_id,'features'].iloc[0]

movie_embedding_df['sim_value'] = movie_embedding_df['features'].apply(lambda x: 1 - distance.cosine(user_embedding, x))

movie_embedding_df = movie_embedding_df.sort_values('sim_value', ascending=False)

print(movie_embedding_df[['movie_id','genres','sim_value']].head(50))

'''查看用户已经点击过的'''
movie_embedding_df = movie_embedding_df[movie_embedding_df['movie_id'].isin(current_track_df['movie_id'].values)]

print(movie_embedding_df[['movie_id','genres','sim_value']].sort_values('sim_value',ascending=False).head(50))

