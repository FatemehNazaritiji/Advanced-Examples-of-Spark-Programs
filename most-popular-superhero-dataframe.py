import logging
from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from typing import Optional


def configure_logging() -> None:
    """
    Configures the logging settings for the script.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_data(
    spark: SparkSession,
    file_path: str,
    schema: Optional[StructType] = None,
    sep: Optional[str] = None,
) -> Optional[DataFrame]:
    """
    Loads data from a file into a DataFrame with the specified schema.

    Args:
    spark (SparkSession): The SparkSession object.
    file_path (str): The path to the file.
    schema (Optional[StructType]): The schema for the DataFrame. Default is None.
    sep (Optional[str]): The separator used in the file. Default is None.

    Returns:
    Optional[DataFrame]: The loaded DataFrame or None if there is an error.
    """
    try:
        if schema:
            df = spark.read.schema(schema).option("sep", sep).csv(file_path)
        else:
            df = spark.read.text(file_path)
        logging.info(f"Data loaded successfully from {file_path}.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None


def get_most_popular_superhero(
    connections: DataFrame, names: DataFrame
) -> Optional[Row]:
    """
    Finds the most popular superhero based on the number of connections.

    Args:
    connections (DataFrame): The DataFrame containing superhero connections.
    names (DataFrame): The DataFrame containing superhero names.

    Returns:
    Optional[Row]: A Row containing the name and number of connections of the most popular superhero.
    """
    try:
        most_popular = connections.sort(func.col("connections").desc()).first()
        most_popular_name = (
            names.filter(func.col("id") == most_popular["id"])
            .select("name")
            .first()
        )
        if most_popular_name and most_popular:
            return Row(
                name=most_popular_name["name"],
                connections=most_popular["connections"],
            )
        else:
            return None
    except Exception as e:
        logging.error(f"Error finding the most popular superhero: {e}")
        return None


def main() -> None:
    """
    Main function to configure Spark, load data, and find the most popular superhero.

    Steps:
    1. Configure logging for the script.
    2. Create a SparkSession.
    3. Define the schema for the Marvel names file.
    4. Load the Marvel names data.
    5. Load the Marvel graph data.
    6. Calculate the number of connections for each superhero.
    7. Find the most popular superhero.
    8. Display the most popular superhero.
    9. Stop the Spark session.
    """
    configure_logging()

    try:
        # Step 2: Create a SparkSession
        spark = SparkSession.builder.appName(
            "MostPopularSuperhero"
        ).getOrCreate()
        logging.info("Spark session created.")
    except Exception as e:
        logging.error(f"Error creating Spark session: {e}")
        return

    try:
        # Step 3: Define the schema for the Marvel names file
        schema = StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("name", StringType(), True),
            ]
        )

        # Step 4: Load the Marvel names data
        names_file_path = "file:///SparkCourse/Advanced Examples of Spark Programs/Marvel-names.txt"
        names_df: Optional[DataFrame] = load_data(
            spark, names_file_path, schema, " "
        )
        if names_df is None:
            raise ValueError("Names DataFrame is None, exiting.")

        # Step 5: Load the Marvel graph data
        graph_file_path = "file:///SparkCourse/Advanced Examples of Spark Programs/Marvel-graph.txt"
        lines_df: Optional[DataFrame] = load_data(spark, graph_file_path)
        if lines_df is None:
            raise ValueError("Graph DataFrame is None, exiting.")

        # Step 6: Calculate the number of connections for each superhero
        connections_df = (
            lines_df.withColumn(
                "id", func.split(func.trim(func.col("value")), " ")[0]
            )
            .withColumn(
                "connections",
                func.size(func.split(func.trim(func.col("value")), " ")) - 1,
            )
            .groupBy("id")
            .agg(func.sum("connections").alias("connections"))
        )

        # Step 7: Find the most popular superhero
        most_popular_superhero = get_most_popular_superhero(
            connections_df, names_df
        )
        if most_popular_superhero is None:
            raise ValueError("Could not determine the most popular superhero.")

        # Step 8: Display the most popular superhero
        logging.info(
            f"{most_popular_superhero['name']} is the most popular superhero"
            f"with {most_popular_superhero['connections']} co-appearances."
        )
    except Exception as e:
        logging.error(f"Error during processing: {e}")
    finally:
        # Step 9: Stop the Spark session
        spark.stop()
        logging.info("Spark session stopped.")


if __name__ == "__main__":
    main()
