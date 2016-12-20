import sys

required_verion = (3,)
if sys.version_info < required_verion:
    raise ValueError('mie-py needs at least python {}! You are trying to install it under python {}'.format('.'.join(str(i) for i in required_verion), sys.version))

import ez_setup
ez_setup.use_setuptools()

from setuptools import setup, find_packages
setup(
    name="mie-py",
    version="0.1",
    packages=find_packages(),
    author="Hagen Telg",
    author_email="hagen@hagnet.net",
    description="This package contains tools to perform mie calculations",
    license="MIT",
    keywords="mie scattering calculations",
    url="http://github.com/hagne/mie-py",
    install_requires=['numpy','pandas'],
    extras_require={'plotting': ['matplotlib'],
                    'testing': ['scipy']},
    test_suite='nose.collector',
    tests_require=['nose'],
)