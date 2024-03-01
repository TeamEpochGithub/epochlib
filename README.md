# The Epochalyst package is the base package for Team Epoch competitions

This package contains many modules and classes necessary to construct the src code for machine learning competitions.

Epochalyst: A fusion of "Epoch" and "Catalyst," this name positions your pipeline as a catalyst in the field of machine learning, sparking significant advancements and transformations.

## Pytest coverage report

To generate pytest coverage report run

```python
python -m pytest --cov=epochalyst --cov-report=html:coverage_re
```

## Documentation

Documentation is generated using [Sphinx](https://www.sphinx-doc.org/en/master/).

To make the documentation, run `make html` with `docs` as the working directory. The documentation can then be found in `docs/_build/html/index.html`.

Here's a short command to make the documentation and open it in the browser:

```shell
cd ./docs/;
./make.bat html; start chrome file://$PWD/_build/html/index.html
cd ../
```
