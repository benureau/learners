import os
from distutils.core import setup

setup(
    name = "learners",
    version = "0.1",
    author = "Fabien Benureau",
    author_email = "fabien.benureau@inria.fr",
    description = ("A collection of simple incremental learning algorithm"),
    license = "Open Science.",
    keywords = "learning algorithm",
    url = "flowers.inria.fr",
    packages=['learners', 'learners.algorithms'],
    classifiers=[],
)
