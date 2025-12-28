# Copyright (c) 2021 Philip May, Deutsche Telekom AG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build script for setuptools."""

import os

import setuptools


project_name = "lazy_imports"
source_code = "https://github.com/bachorp/lazy-imports"
keywords = "import imports lazy"
install_requires: list[str] = []
extras_require = {
    "checking": [
        "black",
        "flake8",
        "isort",
        "mdformat",
        "pydocstyle",
        "mypy",
        "pylint",
        "pylintfileheader",
    ],
    "testing": ["pytest", "packaging"],
}
extras_require["all"] = list({package_name for value in extras_require.values() for package_name in value})


def get_version():
    """Read version from ``__init__.py``."""
    version_filepath = os.path.join(os.path.dirname(__file__), project_name, "__init__.py")
    with open(version_filepath) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
    assert False


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name=project_name,
    version=get_version(),
    maintainer="Pascal Bachor",
    author="Pascal Bachor",
    author_email="lazy-imports.vista851@passmail.net",
    description="Tool to support lazy imports",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Changelog": source_code + "/blob/HEAD/CHANGELOG.md",
        "Bug Tracker": source_code + "/issues",
        "Source Code": source_code,
    },
    packages=["lazy_imports", "lazy_imports.v0"],
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    keywords=keywords,
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
)
