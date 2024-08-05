import logging
import sys
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as func
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    LongType,
)
from typing import Tuple


def configure_logging() -> None:
    """Configures the logging settings for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def create_spark_session(app_name: str) -> SparkSession:
    """Creates and returns a SparkSession object.

    Args:
        app_name (str): The name of the Spark application.

    Returns:
        SparkSession: The SparkSession object initialized with the given app name.
    """
    return (
        SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()
    )


def compute_cosine_similarity(data: DataFrame) -> DataFrame:
    """Computes the cosine similarity between movie pairs.

    This function calculates the cosine similarity between pairs of movies based on their ratings
    and returns a DataFrame with the similarity scores.

    Args:
        data (DataFrame): A DataFrame containing pairs of movie ratings.

    Returns:
        DataFrame: A DataFrame containing movie pairs, similarity scores, and number of rating pairs.
    """
    pair_scores = (
        data.withColumn("xx", func.col("rating1") * func.col("rating1"))
        .withColumn("yy", func.col("rating2") * func.col("rating2"))
        .withColumn("xy", func.col("rating1") * func.col("rating2"))
    )

    calculate_similarity = pair_scores.groupBy("movie1", "movie2").agg(
        func.sum(func.col("xy")).alias("numerator"),
        (
            func.sqrt(func.sum(func.col("xx")))
            * func.sqrt(func.sum(func.col("yy")))
        ).alias("denominator"),
        func.count(func.col("xy")).alias("numPairs"),
    )

    # Round the score to 2 decimal places
    result = calculate_similarity.withColumn(
        "score",
        func.when(
            func.col("denominator") != 0,
            func.round(func.col("numerator") / func.col("denominator"), 5),
        ).otherwise(0),
    ).select("movie1", "movie2", "score", "numPairs")

    return result


def get_movie_name(movie_names: DataFrame, movie_id: int) -> str:
    """Retrieves the name of a movie given its ID.

    Args:
        movie_names (DataFrame): A DataFrame containing movie IDs and names.
        movie_id (int): The ID of the movie.

    Returns:
        str: The name of the movie.
    """
    try:
        result = (
            movie_names.filter(func.col("movieID") == movie_id)
            .select("movieTitle")
            .collect()[0]
        )
        return result[0]
    except IndexError:
        logging.error(f"Movie ID {movie_id} not found in the dataset.")
        return "Unknown"


def load_data(spark: SparkSession) -> Tuple[DataFrame, DataFrame]:
    """Loads movie names and movie ratings data.

    Args:
        spark (SparkSession): The SparkSession object.

    Returns:
        Tuple[DataFrame, DataFrame]: Two DataFrames, one for movie names and one for movie ratings.
    """
    movie_names_schema = StructType(
        [
            StructField("movieID", IntegerType(), True),
            StructField("movieTitle", StringType(), True),
        ]
    )

    movies_schema = StructType(
        [
            StructField("userID", IntegerType(), True),
            StructField("movieID", IntegerType(), True),
            StructField("rating", IntegerType(), True),
            StructField("timestamp", LongType(), True),
        ]
    )

    movie_names = (
        spark.read.option("sep", "|")
        .option("charset", "ISO-8859-1")
        .schema(movie_names_schema)
        .csv(
            "file:///SparkCourse/Advanced Examples of Spark Programs/ml-100k/u.item"
        )
    )

    movies = (
        spark.read.option("sep", "\t")
        .schema(movies_schema)
        .csv(
            "file:///SparkCourse/Advanced Examples of Spark Programs/ml-100k/u.data"
        )
    )

    return movie_names, movies


def main(movie_id: int) -> None:
    """Main function to compute and display top similar movies based on cosine similarity.

    Steps:
    1. Configure logging for the script.
    2. Create a SparkSession.
    3. Load movie data (names and ratings).
    4. Compute movie pair similarities.
    5. Filter and display the top similar movies for the given movie ID.

    Args:
        movie_id (int): The ID of the movie to find similarities for.
    """
    configure_logging()

    try:
        # Step 2: Create a SparkSession
        spark = create_spark_session("MovieSimilarities")
        logging.info("Spark session created.")

        # Step 3: Load movie data
        movie_names, movies = load_data(spark)
        logging.info("Movie data loaded.")

        # Prepare the ratings data
        ratings = movies.select("userID", "movieID", "rating")

        # Step 4: Compute movie pair similarities
        movie_pairs = (
            ratings.alias("ratings1")
            .join(
                ratings.alias("ratings2"),
                (func.col("ratings1.userID") == func.col("ratings2.userID"))
                & (
                    func.col("ratings1.movieID") < func.col("ratings2.movieID")
                ),
            )
            .select(
                func.col("ratings1.movieID").alias("movie1"),
                func.col("ratings2.movieID").alias("movie2"),
                func.col("ratings1.rating").alias("rating1"),
                func.col("ratings2.rating").alias("rating2"),
            )
        )

        movie_pair_similarities = compute_cosine_similarity(
            movie_pairs
        ).cache()
        logging.info("Computed cosine similarity for movie pairs.")

        # Step 5: Filter and display top similar movies
        score_threshold = 0.97
        co_occurrence_threshold = 50.0

        filtered_results = movie_pair_similarities.filter(
            (
                (func.col("movie1") == movie_id)
                | (func.col("movie2") == movie_id)
            )
            & (func.col("score") > score_threshold)
            & (func.col("numPairs") > co_occurrence_threshold)
        )

        results = filtered_results.sort(func.col("score").desc()).take(10)
        print()
        logging.info(
            f"Top 10 similar movies for {get_movie_name(movie_names, movie_id)}:"
        )
        logging.info(f"{'Movie Title':<50} {'Score':<6} {'Strength':<8}")
        for result in results:
            similar_movie_id = (
                result.movie1 if result.movie1 != movie_id else result.movie2
            )
            similar_movie_name = get_movie_name(movie_names, similar_movie_id)
            logging.info(
                f"{similar_movie_name:<50} {result.score:<6.5f} {result.numPairs:<12}"
            )
        print()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if "spark" in locals():
            spark.stop()
            logging.info("Spark session stopped.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            movieID = int(sys.argv[1])
            main(movieID)
        except ValueError:
            logging.error("Please provide a valid movie ID as an integer.")
    else:
        logging.error("Movie ID argument missing. Please provide a movie ID.")
