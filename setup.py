try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='storm_forcast',
      version='2.0',
      description="ACDS - Katrina",
      long_description="ACDS - The day after tomorrow - Katrina",
      url='https://github.com/ese-msc-2021',
      author="Imperial College London",
      author_email='bo.pang21@imperial.ac.uk',
      packages=['storm_forcast'])
