from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='opponent_process_ploscompbiol2020',
      version='1.0',
      description='Visual opponent process for navigation',
      long_description=readme(),
      url='https://github.com/antnavigationteam/opponent_process_ploscompbiol2020',
      author='Ant Navigation Team',
      license='BSD',
      packages=find_packages()
      )
