import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, IntegerType, LongType
import codecs
from typing import Dict, Optional


def configure_logging() -> None:
    """
    Configures the logging settings for the script.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_movie_names(file_path: str) -> Dict[int, str]:
    """
    Loads movie names from a file and returns a dictionary of movieID to movieName.

    Args:
    file_path (str): The path to the u.item file.

    Returns:
    Dict[int, str]: A dictionary mapping movieID to movieName.
    """
    movie_names = {}
    try:
        with codecs.open(
            file_path, "r", encoding="ISO-8859-1", errors="ignore"
        ) as f:
            for line in f:
                fields = line.split("|")
                movie_names[int(fields[0])] = fields[1]
        logging.info("Movie names loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading movie names: {e}")
    return movie_names


def load_data(
    spark: SparkSession,
    file_path: str,
    schema: StructType,
) -> Optional[DataFrame]:
    """
    Loads data from a CSV file into a DataFrame with the specified schema.

    Args:
    spark (SparkSession): The SparkSession object.
    file_path (str): The path to the CSV file.
    schema (StructType): The schema for the DataFrame.

    Returns:
    Optional[DataFrame]: The loaded DataFrame or None if there is an error.
    """
    try:
        df = spark.read.option("sep", "\t").schema(schema).csv(file_path)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None


def main() -> None:
    """
    Main function to configure Spark, load data, and analyze movie ratings.

    Steps:
    1. Configure logging for the script.
    2. Load movie names and broadcast them.
    3. Create a SparkSession.
    4. Define the schema for the u.data file.
    5. Load the movie data.
    6. Group by movieID and count occurrences.
    7. Add movie titles using a UDF.
    8. Sort the results by count.
    9. Display the top 10 movies.
    10. Stop the Spark session.
    """
    configure_logging()

    # Step 1: Load movie names
    movie_names_file = (
        "C:/SparkCourse/Advanced Examples of Spark Programs/ml-100k/u.item"
    )
    movie_names = load_movie_names(movie_names_file)

    try:
        # Step 2: Create a SparkSession
        spark = SparkSession.builder.appName("PopularMovies").getOrCreate()
        logging.info("Spark session created.")
    except Exception as e:
        logging.error(f"Error creating Spark session: {e}")
        return

    try:
        # Step 3: Broadcast movie names
        name_dict = spark.sparkContext.broadcast(movie_names)

        # Step 4: Define the schema for the u.data file
        schema = StructType(
            [
                StructField("userID", IntegerType(), True),
                StructField("movieID", IntegerType(), True),
                StructField("rating", IntegerType(), True),
                StructField("timestamp", LongType(), True),
            ]
        )

        # Step 5: Load the movie data
        file_path = "file:///SparkCourse/Advanced Examples of Spark Programs/ml-100k/u.data"
        movies_df: Optional[DataFrame] = load_data(spark, file_path, schema)
        if movies_df is None:
            raise ValueError("DataFrame is None, exiting.")
    except Exception as e:
        logging.error(f"Error during data loading and validation: {e}")
        return

    try:
        # Step 6: Group by movieID and count occurrences
        movie_counts = movies_df.groupBy("movieID").count()

        # Step 7: Create a UDF to look up movie names from the broadcasted dictionary
        lookup_name_udf = func.udf(lambda movieID: name_dict.value[movieID])

        # Add a movieTitle column using the UDF
        movies_with_names = movie_counts.withColumn(
            "movieTitle", lookup_name_udf(func.col("movieID"))
        )

        # Step 8: Sort the results by count
        sorted_movies_with_names = movies_with_names.orderBy(
            func.desc("count")
        )

        # Step 9: Display the top 10 movies
        logging.info("Displaying the top 10 movies:")
        sorted_movies_with_names.show(10, False)
    except Exception as e:
        logging.error(f"Error during DataFrame operations: {e}")
    finally:
        # Step 10: Stop the Spark session
        spark.stop()
        logging.info("Spark session stopped.")


if __name__ == "__main__":
    main()
