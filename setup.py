import os
from setuptools import setup

import versioneer

setup(
    name         = "learners",
    version      = "1.0.2",
    cmdclass     = versioneer.get_cmdclass(),
    author       = "Fabien Benureau",
    author_email = "fabien.benureau@inria.fr",
    url          = 'github.com/humm/learners.git',
    download_url = 'https://github.com/humm/learners/tarball/1.0',
    maintainer   = 'Fabien Benureau',
    description  = "A collection of simple incremental forward and inverse model learning algorithms",
    license      = "Open Science License (see fabien.benureau.com/openscience.html)",
    keywords     = "learning algorithm",
    packages     = ['learners', 'learners.algorithms'],
    install_requires=['scicfg','numpy', 'scipy', 'scikit-learn'],
    dependency_links=[
        "https://github.com/flowersteam/scicfg/tarball/master#egg=scicfg-1.0",
    ],

    # in order to avoid 'zipimport.ZipImportError: bad local file header'
    zip_safe=False,
)
