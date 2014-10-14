import os
from setuptools import setup

setup(
    name = "learners",
    version = "1.0",
    author = "Fabien Benureau",
    author_email = "fabien.benureau@inria.fr",
    description = ("A collection of simple incremental learning algorithm"),
    license = "Open Science (see fabien.benureau.com/openscience.html)",
    keywords = "learning algorithm",
    url = "flowers.inria.fr",
    packages=['learners', 'learners.algorithms'],
)
