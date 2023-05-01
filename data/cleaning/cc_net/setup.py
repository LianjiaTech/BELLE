# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from setuptools import setup  # type: ignore

setup(
    name="cc_net",
    version="1.0.0",
    packages=["cc_net"],
    # metadata to display on PyPI
    author="Guillaume Wenzek",
    author_email="guw@fb.com",
    description="Tools to download and clean Common Crawl",
    keywords="common crawl dataset",
    url="https://github.com/facebookresearch/cc_net",
    license="CC-BY-NC-4.0",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/facebookresearch/cc_net/issues",
        "Source Code": "https://github.com/facebookresearch/cc_net",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires=">=3.7",
    install_requires=[
        "beautifulsoup4>=4.7.1",
        "pandas>=0.23.4",
        "requests>=2.22.0",
        "fasttext>=0.9.1",
        "sentencepiece>=0.1.82",
        "kenlm @ git+https://github.com/kpu/kenlm.git@master",
        "func_argparse>=1.1.1",
        "psutil>=5.6.3",
        "sacremoses",
        "submitit>=1.0.0",
        "typing_extensions",
    ],
    extras_require={
        "dev": ["mypy==0.790", "pytest", "black==19.3b0", "isort==5.6.4"],
        # To use scripts inside cc_net/tools
        "tools": ["lxml", "sentence_splitter"],
        # Memory-efficient hashset.
        # This fork only compiles the kind of dict used by cc_net.
        # Full version is at https://github.com/atom-moyer/getpy
        "getpy": ["getpy @ git+https://github.com/gwenzek/getpy.git@v0.9.10-subset"],
    },
    package_data={"cc_net": ["data/*"]},
)
