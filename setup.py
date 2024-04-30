from setuptools import find_packages, setup

setup(
    name="continunet",
    packages=find_packages(include=["continunet"]),
    version="0.0.0",
    description="Source finding package for radio continuum data.",
    author="Hattie Stewart",
    install_requires=[],
    test_suite="continunet/tests",
)
