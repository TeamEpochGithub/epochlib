# Caching

Caching is an important aspect of machine learning competitions, whether it is speeding up your training, development or being able to recover from failures there are many use cases. In this folder an interface is defined for what methods a cacher should implement such that it can be injected into other objects. Cachers should output the same data type when reading the data as that they received when storing. In the name a cacher should therefore specify two things: the storage type like parquet and the output data type like a numpy array.

## Available cachers

A list of the available cachers is provided below:
