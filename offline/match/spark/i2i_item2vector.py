import os

os.environ['SPARK_HOME'] = 'F:\spark\spark-2.4.3-bin-hadoop2.7'
from pyspark import SparkConf
from pyspark.sql import SparkSession
from util.config import PROJECT_PATH
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2Vec


'''用户点击记录：根据时间进行排序'''
class UDFFuctions():

    @staticmethod
    def sortF(movie_list, time_list):
        pairs = []
        for m, t in zip(movie_list, time_list):
            pairs.append((m, t))

        pairs = sorted(pairs, key=lambda x: x[1])
        return [x[0] for x in pairs]


path = PROJECT_PATH + 'data/ratings.dat'

conf = SparkConf().setAppName('i2i_als').setMaster('local')

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
    df = df.withColumn(name, split('value', '::').getItem(i).astype("int"))

print(df.printSchema)
'''dataframe修改数据类型'''
df = df.withColumn("score", df['score'].astype("float"))

df = df.drop('value')

'''
使用agg 针对dataframe 做聚合操作
avg：取平均
withColumnRenamed：重命名 或者直接使用 alias
collect 返回集合row类型，根据字段名取值即可
'''
score_mean = df.agg(F.avg(df['score']).alias('score_avg')). \
    collect()[0]['score_avg']

print(score_mean)
df = df.where(df['score'] > score_mean)

'''自定义udf函数：函数，返回类型'''
sortUdf = udf(UDFFuctions.sortF, ArrayType(StringType()))

'''
1.过滤低于平均值的打分记录
2.聚合用户所有的行为记录，并以时间进行排序
'''
df = df.where(df['score'] > score_mean). \
    groupby("user_id"). \
    agg(sortUdf(F.collect_list('movie_id'), F.collect_list('time')).alias('movie_ids'))
df.show()

'''
这里可能需要调节driver的内存大小，numPartitions默认为1，如果数据量比较大时，需要设置大一点
'''
word2vec = Word2Vec(minCount=3, windowSize=5, inputCol='movie_ids', outputCol='movie_2vec')
model = word2vec.fit(df)

model.getVectors().show(10, truncate=False)

model.getVectors().select('word','vector').toPandas().to_csv('i2i_movie_embedding.csv')

