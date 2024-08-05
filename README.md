# Advanced-Examples-of-Spark-Programs

This repository contains several projects developed while taking the Udemy course ["Taming Big Data with Apache Spark - Hands On!"](https://www.udemy.com/course/taming-big-data-with-apache-spark-hands-on/?couponCode=KEEPLEARNING). Each script utilizes Apache Spark to perform data processing tasks ranging from basic data manipulations to complex analytics. Below is an overview of each script and the key learnings from the projects.

## Projects Overview

### 1. Degrees of Separation

- **Script:** `degrees-of-separation.py`
- **Description:** This script uses the Breadth-First Search (BFS) algorithm to find the degrees of separation between two Marvel characters, represented as nodes in a graph. The idea is to explore how many steps it takes to link one character to another through their appearances in the same comic.
- **Key Learnings:** Learned how to implement BFS in Spark, handle graph data, and utilize accumulators to manage state across distributed computations.

### 2. Most Popular Superhero

- **Script:** `most-popular-superhero-dataframe.py`
- **Description:** This script identifies the most popular superhero in the Marvel dataset based on the number of comic book appearances. It uses Spark DataFrames to aggregate and count appearances.
- **Key Learnings:** Gained experience with Spark DataFrames, including data aggregation and sorting operations. Enhanced understanding of how to work with complex datasets and extract meaningful insights.

### 3. Movie Similarities

- **Script:** `movie-similarities-dataframe.py`
- **Description:** This script calculates movie similarities based on user ratings using the cosine similarity metric. It aims to recommend movies that are similar to a given movie based on user preferences.
- **Key Learnings:** Developed skills in pairwise data manipulations, learned to calculate cosine similarities, and practiced filtering and sorting large datasets in Spark.

### 4. Popular Movies Nicely Formatted

- **Script:** `popular-movies-nice-dataframe.py`
- **Description:** This script processes movie rating data to find and display the most popular movies. The output is nicely formatted to improve readability.
- **Key Learnings:** Enhanced skills in data formatting and presentation using Spark. Learned how to apply formatting techniques to improve the visual presentation of the output data.

## Setup and Running the Scripts

To run these scripts, you will need to set up Apache Spark on your local machine or a cluster. Each script can be submitted to the Spark cluster using the following command:

```bash
spark-submit <script_name>.py
```

For Movie Similarities project ONLY use:
```bash
spark-submit movie-similarities-dataframe.py movieID
```
