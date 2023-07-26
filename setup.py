from setuptools import setup
import codecs
import os.path


# keeping this code for now, but it is not used
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

# pyproject should be used instead of setup.py
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# now that pyproject.toml is used, we should only need a minimal setup.py
# to ensure compatibility with older versions of pip
# see https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html
# the full setup.py is commented out below, will be removed in future.
setup()

# setup(
#     name='pgmuvi',
#     version=get_version("pgmuvi/__init__.py"),    
#     description='A package for GP inference of multiwavelength variability',
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     url='https://github.com/icsm/pgmuvi',
#     project_urls={
#         "Bug Tracker": "https://github.com/icsm/pgmuvi/issues",
#     },

#     author='Peter Scicluna, Kathryn Jones, Stefan Waterval',
#     author_email='peter.scicluna@eso.org, kathryn.jones@unibe.ch, sw4445@nyu.edu',
#     license='GPL',
#     packages=['pgmuvi'],
#     install_requires=['numpy',
#                       'matplotlib',
#                       'seaborn',
#                       'torch',
#                       'gpytorch',
#                       'pyro-ppl',
#                       'arviz',
#                       'xarray',
#                       'tqdm'
#                       ],
#     python_requires=">=3.8", 

#     classifiers=[
#         'Development Status :: 3 - Alpha',
#         'Intended Audience :: Science/Research',
#         'License :: OSI Approved :: GNU Public License v3.0',
#         'Operating System :: POSIX :: Linux',
#         'Programming Language :: Python :: 3',
#     ],
# )
