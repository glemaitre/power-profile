from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'scipy'
    ]

setup(name='scikit-cycling',
      version='0.1',
      description='Python module for cycling tools.',
      classifiers=[
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.4",
          ],
      author='Cedric Lemaitre, Guillaume Lemaitre',
      author_email='c.lemaitre58@gmail.com, g.lemaitre58@gmail.com',
      url='https://github.com/glemaitre/power-profile',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      )
