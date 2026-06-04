# main.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# Load datasets
spark = SparkSession.builder.appName("MusicAnalysis").getOrCreate()

df_listening_logs = spark.read.csv('listening_logs.csv', header=True, inferSchema=True)
df_songs_metadata = spark.read.csv('songs_metadata.csv', header=True, inferSchema=True)

df = df_listening_logs.join(df_songs_metadata, on="song_id", how="inner")

# Task 1: User Favorite Genres

# Count how many times each user listens to each genre
genre_counts = df.groupBy("user_id", "genre") \
    .count()

# Find the most listened genre per user by sorting
user_favorite_genres = genre_counts \
    .orderBy("user_id", col("count").desc())

# Keep only the top genre per user
user_favorite_genres = user_favorite_genres.dropDuplicates(["user_id"])

user_favorite_genres.write.mode("overwrite").csv("outputs/task1_favorite_genres")

# Task 2: Average Listen Time
avg_listen_time = df.groupBy("user_id") \
    .agg(avg("duration_sec").alias("average_listening_time"))

avg_listen_time.write.mode("overwrite").csv("outputs/task2_avg_listen_time")


# Task 3: Create your own Genre Loyalty Scores and rank them and list out top 10

# Total listens per user
total_listens = df.groupBy("user_id").count() \
    .withColumnRenamed("count", "total_listens")

# Get the top genre per user by sorting and removing duplicates
top_genre = genre_counts.orderBy(col("count").desc()) \
    .dropDuplicates(["user_id"]) \
    .select(
        "user_id",
        col("genre").alias("top_genre"),
        col("count").alias("top_genre_count")
    )

# Join totals with top genre
loyalty = total_listens.join(top_genre, "user_id")

# Create loyalty score
loyalty = loyalty.withColumn(
    "loyalty_score",
    col("top_genre_count") / col("total_listens")
)

# Get top 10 users
top10_loyal_users = loyalty.orderBy(col("loyalty_score").desc()).limit(10)

# Save output
top10_loyal_users.write.mode("overwrite").csv("outputs/task3_loyalty_top10")

# Task 4: Identify users who listen between 12 AM and 5 AM
night_users = df.withColumn("hour", hour(col("timestamp"))) \
    .filter((col("hour") >= 0) & (col("hour") < 5)) \
    .select("user_id").distinct()

night_users.write.mode("overwrite").csv("outputs/task4_night_owls")