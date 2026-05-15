from setuptools import setup, find_packages

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

setup(name='sqwlib',
      version='1.0',
      description='Szegedy Quantum Walk Simulator',
      author='Sergio A. Ortega and Daniel K. Park',
      author_email='s.a.ortega@yonsei.ac.kr',
      url='https://github.com/...',
      license='Apache 2.0',
      # classifiers=[
      #   "Environment :: Console",
      #   "License :: OSI Approved :: Apache Software License",
      #   "Intended Audience :: Developers",
      #   "Intended Audience :: Science/Research",
      #   "Operating System :: Microsoft :: Windows",
      #   "Operating System :: MacOS",
      #   "Operating System :: POSIX :: Linux",
      #   "Programming Language :: Python :: 3 :: Only",
      #   "Topic :: Scientific/Engineering",
      # ],
      install_requires=REQUIREMENTS,
      python_requires=">=3.0",
      include_package_data=True,
      packages=find_packages(),
     )
