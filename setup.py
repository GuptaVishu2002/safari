from setuptools import setup, find_packages

setup(
    name="safari",
    version="0.1.0",
    description="Forked research code from HazyResearch/safari",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "einops",
        "opt_einsum",
    ],
)