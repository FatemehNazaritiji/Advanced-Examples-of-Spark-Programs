import logging
from pyspark import SparkConf, SparkContext
from typing import List, Tuple


def configure_logging() -> None:
    """Configures the logging settings for the script.

    Sets up logging with INFO level and a specified format for log messages.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def initialize_spark(app_name: str) -> SparkContext:
    """Initializes Spark configuration and context.

    Args:
        app_name (str): The name of the Spark application.

    Returns:
        SparkContext: The SparkContext object initialized with the given app name.
    """
    conf = SparkConf().setMaster("local").setAppName(app_name)
    return SparkContext(conf=conf)


# The characters we wish to find the degree of separation between
startCharacterID = 5306  # SpiderMan
targetCharacterID = 14  # ADAM 3,031 (target character)

# Accumulator to signal when the target character is found during BFS traversal
hitCounter = None


def convert_to_bfs(line: str) -> Tuple[int, Tuple[List[int], int, str]]:
    """Converts a line of text from the input file into a BFS node.

    The input file line format is:
    heroID connection1 connection2 connection3 ...

    Each line represents a hero and their connections to other heroes (edges).

    Args:
        line (str): A line of text from the input file.

    Returns:
        Tuple[int, Tuple[List[int], int, str]]: A tuple representing the hero ID,
        list of connections, initial distance, and color.

    Color Legend:
        WHITE: Unexplored nodes that have not been visited.
        GRAY: Nodes that are being explored (discovered but not fully processed).
        BLACK: Nodes that have been fully explored.
    """
    fields = line.split()
    heroID = int(fields[0])
    connections = [int(connection) for connection in fields[1:]]

    color = "WHITE"
    distance = 9999

    # If this is the starting character, set distance to 0 and color to GRAY
    if heroID == startCharacterID:
        color = "GRAY"
        distance = 0

    return (heroID, (connections, distance, color))


def create_starting_rdd(sc: SparkContext) -> None:
    """Loads the initial graph data and converts it into a format suitable for BFS.

    Reads the graph data from a text file and maps each line to a BFS node using
    the convert_to_bfs function.

    Args:
        sc (SparkContext): The SparkContext object.

    Returns:
        RDD: An RDD containing the initial graph data prepared for BFS.
    """
    inputFile = sc.textFile(
        "file:///sparkcourse/Advanced Examples of Spark Programs/marvel-graph.txt"
    )
    return inputFile.map(convert_to_bfs)


def bfs_map(
    node: Tuple[int, Tuple[List[int], int, str]]
) -> List[Tuple[int, Tuple[List[int], int, str]]]:
    """Expands a GRAY node by generating new nodes for its connections.

    If a node is GRAY, it means we are currently exploring it and should check
    its connections. For each connection, a new node is created with an incremented distance.

    If the target character is found, the accumulator is updated to signal that the
    search is complete. The original node is also emitted, marked as BLACK, to indicate
    that it has been fully processed.

    Args:
        node (Tuple[int, Tuple[List[int], int, str]]): A tuple representing the current node.

    Returns:
        List[Tuple[int, Tuple[List[int], int, str]]]: A list of nodes resulting from the expansion.
    """
    characterID, (connections, distance, color) = node

    results = []

    # If this node needs to be expanded...
    if color == "GRAY":
        for connection in connections:
            newCharacterID = connection
            newDistance = distance + 1
            newColor = "GRAY"
            if targetCharacterID == connection:
                # Increment accumulator if target character is found
                hitCounter.add(1)

            # Create a new entry for each connection
            newEntry = (newCharacterID, ([], newDistance, newColor))
            results.append(newEntry)

        # We've processed this node, so color it BLACK
        color = "BLACK"

    # Emit the input node so we don't lose it.
    results.append((characterID, (connections, distance, color)))
    return results


def bfs_reduce(
    data1: Tuple[List[int], int, str], data2: Tuple[List[int], int, str]
) -> Tuple[List[int], int, str]:
    """Reduces two nodes by preserving the shortest distance, darkest color, and combining edges.

    During the reduce phase, multiple pieces of information for the same node ID may be encountered.
    This function resolves these by:
    - Combining the edges (connections).
    - Preserving the minimum distance found.
    - Preserving the darkest (most processed) color found.

    Args:
        data1 (Tuple[List[int], int, str]): The first node's data.
        data2 (Tuple[List[int], int, str]): The second node's data.

    Returns:
        Tuple[List[int], int, str]: A tuple representing the merged node data.
    """
    edges1, distance1, color1 = data1
    edges2, distance2, color2 = data2

    # Initialize defaults for distance, color, and edges
    distance = min(distance1, distance2)
    color = color1 if color1 in ("GRAY", "BLACK") else color2
    edges = edges1 if edges1 else edges2

    return (edges, distance, color)


def main() -> None:
    """Main function to perform the BFS traversal to find the degrees of separation
    between the start character and target character.

    Steps:
    1. Configure logging for the script.
    2. Initialize Spark context.
    3. Create an RDD with the initial graph data prepared for BFS.
    4. Perform BFS iterations.
    5. Print results when the target character is found.
    """
    configure_logging()
    global hitCounter

    try:
        # Step 2: Initialize Spark context
        sc = initialize_spark("DegreesOfSeparation")
        hitCounter = sc.accumulator(0)
        logging.info("Spark context initialized.")

        # Step 3: Create an RDD with the initial graph data
        iterationRdd = create_starting_rdd(sc)

        # Step 4: Perform up to 10 iterations of BFS
        for iteration in range(0, 10):
            logging.info(f"Running BFS iteration# {iteration + 1}")

            # Expand GRAY nodes and create new vertices as needed
            mapped = iterationRdd.flatMap(bfs_map)

            # Evaluate the RDD, updating the accumulator in the process
            logging.info(f"Processing {mapped.count()} values.")

            # Check if the target character was found
            if hitCounter.value > 0:
                logging.info(
                    f"Hit the target character! From {hitCounter.value} different direction(s)."
                )
                break

            # Reduce stage combines data for each character ID, preserving the shortest path and darkest color
            iterationRdd = mapped.reduceByKey(bfs_reduce)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if "sc" in locals():
            sc.stop()
            logging.info("Spark context stopped.")


if __name__ == "__main__":
    main()
