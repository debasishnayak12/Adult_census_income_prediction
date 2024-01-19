from setuptools import find_packages,setup


setup(
    name='INCOMEPREDICTION',
    version='0.0.1',
    author='Debasish',
    author_email='nayakdebasish7205@gmail.com',
    install_requires=["scikit-learn","pandas","numpy"],
    packages=find_packages()
)