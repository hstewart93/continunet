from setuptools import find_packages, setup

setup(
    name="continunet",
    packages=find_packages(include=["continunet"]),
    version="0.0.1",
    description="Source finding package for radio continuum data.",
    author="Hattie Stewart",
    install_requires=[],
    setup_requires=["pytest-runner"],
    tests_require=["pytest==4.4.1"],
    test_suite="continunet/tests",
)
