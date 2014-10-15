import os
from setuptools import setup

setup(
    name = "learners",
    version = "1.0",
    author = "Fabien Benureau",
    author_email = "fabien.benureau@inria.fr",
    url='flowers.inria.fr',
    download_url='https://github.com/humm/learners/tarball/1.0',
    maintainer='Fabien Benureau',
    description = "A collection of simple incremental forward and inverse model learning algorithms",
    license = "Open Science License (see fabien.benureau.com/openscience.html)",
    keywords = "learning algorithm",
    packages=['learners', 'learners.algorithms'],
    install_requires=['forest','numpy'],
    dependency_links=[
        "https://github.com/flowersteam/forest/tarball/master#egg=forest-1.0",
    ],
)
