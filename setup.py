from setuptools import find_packages, setup

setup(
    name='ml_tutorials',
    packages=find_packages('src'),
    package_dir={'':'src'},
    description='Tutorials for learning the basics of ML',
    author='Milind Ganjoo'
)
