# Contribution guide

## Overview

COMET is an Open Source toolkit aimed to develop state of the art models that can act as MT evaluation metrics. While we do welcome contributions, in order to guarantee their quality and usefulness, it is necessary that we follow basic guidelines in order to ease development, collaboration and readability.

## Basic guidelines

* The project must fully support Python 3.6 or further.
* Code formatting must stick to the Facebook style, 80 columns and single quotes. Please make sure you have [black](https://github.com/ambv/black) installed and to run it before submitting changes.
* Imports are sorted with [isort](https://github.com/timothycrosley/isort).
* Filenames must be in lowercase.
* Tests are running with [unittest](https://docs.python.org/3/library/unittest.html). Unittest implements a standard test discovery which means that it will search for `test_*.py` files. We do not enforce a minimum code coverage but it is preferrable to have even very basic tests running for critical pieces of code. Always test functions that takes/returns tensor argument to document the sizes.
* The `comet` folder contains core features.

## Contributing

* Keep track of everything by creating issues and editing them with reference to the code! Explain succinctly the problem you are trying to solve and your solution.
* Contributions to `master` should be made through github pull-requests.
* Work in a clean environment (`virtualenv` is nice). 
* Your commit message must start with an infinitive verb (Add, Fix, Remove, ...).
* If your change is based on a paper, please include a clear comment and reference in the code and in the related issue.
* In order to test your local changes, install COMET following the instructions on the [documentation](https://unbabel.github.io/COMET/html/index.html)
