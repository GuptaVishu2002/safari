from setuptools import setup, find_packages

setup(
    name="safari",
    version="0.1.0",
    description="Forked research code from HazyResearch/safari",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
)
