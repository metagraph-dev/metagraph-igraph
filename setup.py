from setuptools import setup, find_packages
import versioneer

setup(
    name="metagraph-igraph",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="python-igraph plugins for Metagraph",
    author="Anaconda, Inc.",
    packages=find_packages(include=["metagraph_igraph", "metagraph_igraph.*"]),
    include_package_data=True,
    install_requires=["metagraph", "python-igraph"],
    entry_points={
        "metagraph.plugins": "plugins=metagraph_igraph.registry:find_plugins"
    },
)
