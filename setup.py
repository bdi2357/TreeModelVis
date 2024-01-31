from setuptools import setup, find_packages

setup(
    name='TreeModelVis',
    version='0.1.0',
    author='Itay Ben Dan',
    author_email='itaybd@gmail.com',
    description='A toolkit for visualizing and customizing tree-based models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/bdi2357/TreeModelVis',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
