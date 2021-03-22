import os

os.environ['SPARK_HOME'] = 'F:\spark\spark-2.4.3-bin-hadoop2.7'
from pyspark import SparkConf
from pyspark.sql import SparkSession
from util.config import PROJECT_PATH
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

path = PROJECT_PATH + 'data/ratings.dat'

conf = SparkConf().setAppName('u2i_als').setMaster('local')

spark = SparkSession. \
    builder. \
    config(conf=conf). \
    getOrCreate()

df = spark.read.text(path)

fields = [('user_id', 0), ('movie_id', 1), ('score', 2), ('time', 3)]

'''
dataframe 做数据切割，并形成新列名
'''
for name, i in fields:
    df = df.withColumn(name, split('value', '::').getItem(i).astype('int'))

'''dataframe修改数据类型'''
df = df.withColumn("score", df['score'].astype("float"))

df = df.drop('value')

print(df.select("user_id").distinct().count())


'''找出每一个用户的打分的均值，然后遍历用户的打分与用户均值比较，大于均值表示喜欢，小于均值表示不喜欢'''
mean_score_df = df.groupby('user_id').agg({"score":"mean"})
mean_score_df.show()
df = df.join(mean_score_df, ['user_id'], how='inner')
df = df.withColumn('label',when(df['score'] > df['avg(score)'],1).otherwise(0) )

df.show()


als = ALS(
          regParam=0.01,
          userCol="user_id",
          itemCol="movie_id",
          ratingCol="score",
          coldStartStrategy="drop")

model = als.fit(df)

model.userFactors.show(5)
model.itemFactors.show(5)

model.userFactors.select('id', 'features').toPandas().to_csv('u2i_user_embedding.csv')

model.itemFactors.select('id', 'features').toPandas().to_csv('u2i_movie_embedding.csv')
