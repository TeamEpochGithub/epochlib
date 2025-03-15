# Caching

Caching is an important aspect of machine learning competitions, whether it is speeding up your training, development or being able to recover from failures there are many use cases. In this folder an interface is defined for what methods a cacher should implement such that it can be injected into other objects. Cachers should output the same data type when reading the data as that they received when storing. In the name a cacher should therefore specify two things: the storage type like parquet and the output data type like a numpy array.

Initially this was implemented to be used with inheritance as then it was thought to be easier to have the access within the class. However,  this hides away a lot of information and is not very usable. By ignoring the field from the hash in a dataclass it can be used without affecting the hash.

## Available cachers

A list of the available cachers is provided below:
.npy:
- Numpy Array
- Dask Array

.parquet:
- Pandas Dataframe
- Dask Dataframe
- Numpy Array
- Dask Array
- Polars Dataframe

.csv:
- Pandas Dataframe
- Dask Dataframe
- Polars Dataframe

.npy_stack:
- Dask Array

.pkl:
- Any Object
