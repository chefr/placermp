import io
import os
from setuptools import find_packages, setup


# From https://github.com/rochacbruno/python-project-template/blob/main/setup.py
def read(*paths, **kwargs):
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


# From https://github.com/rochacbruno/python-project-template/blob/main/setup.py
def read_requirements(path):
    return [line.strip() for line in read(path).split("\n")
            if not line.startswith(('"', "#", "-", "git+"))]


# Template from https://github.com/rochacbruno/python-project-template/blob/main/setup.py
setup(
    name="placermp",
    version="0.1.0",
    description="project_description",
    url="https://github.com/chefr/placermp/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="chefr",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["project_name = project_name.__main__:main"]
    },
    extras_require={},
)
