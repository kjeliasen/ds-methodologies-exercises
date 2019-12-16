import pyspark

spark = pyspark.sql.SparkSession.builder.getOrCreate()


import multiprocessing
import pyspark

# nprocs = multiprocessing.cpu_count()

# spark = (pyspark.sql.SparkSession.builder
#  .master('local')
#  .config('spark.jars.packages', 'mysql:mysql-connector-java:8.0.16')
#  .config('spark.driver.memory', '4G')
#  .config('spark.driver.cores', nprocs)
#  .config('spark.sql.shuffle.partitions', nprocs)
#  .appName('MySparkApplication')
#  .getOrCreate())


def spark_check():
    df = spark.range(10)
    evens_filter = df.id % 2 ==0
    increment_filter = (df.id +1).alias('id')
    print(type(df.select(increment_filter).where(evens_filter)))
    df.select(increment_filter).where(evens_filter).show()


if __name__ == '__main__':
    spark_check()