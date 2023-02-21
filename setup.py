from setuptools import setup, find_packages

packages_found = find_packages()
print('Found packages:')
print(packages_found)

setup(
    name='usefulml',
    version='1.0',
    url='https://github.com/asalomatov/ML',
    author='Andrei Salomatov',
    author_email='andrei.salomatov@gmail.com',
    py_modules=find_packages(),
)

