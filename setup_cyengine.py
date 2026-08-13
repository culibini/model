from Cython.Build import cythonize
from setuptools import Extension, setup

common = {
    "extra_compile_args": ["-O3", "-ffp-contract=off"],
}
omp = {
    "extra_compile_args": ["-O3", "-ffp-contract=off", "-fopenmp"],
    "extra_link_args": ["-fopenmp"],
}

extensions = [
    Extension("cyengine.rng", ["cyengine/rng.pyx"], **common),
    Extension("cyengine.danger", ["cyengine/danger.pyx"], language="c++", **omp),
    Extension("cyengine.precompute", ["cyengine/precompute.pyx"], **omp),
    Extension("cyengine.search", ["cyengine/search.pyx"], **common),
    Extension("cyengine.optimize", ["cyengine/optimize.pyx"], **common),
]

setup(
    name="cyengine",
    ext_modules=cythonize(extensions, language_level=3),
)
