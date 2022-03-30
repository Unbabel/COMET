.. _installation:
Installation
============

Please note that since Python>=3.5 is required, all of the following commands, especially `pip`,
also have to be the Python 3 compliant versions thereof. This might require that you run `pip3` instead.


Install COMET as a package with pip::

   pip install --upgrade pip  # ensures that pip is current 
   pip install unbabel-comet

or::

   pip install unbabel-comet==1.1.0 --use-feature=2020-resolver

Inside your project you can now::

   import comet

