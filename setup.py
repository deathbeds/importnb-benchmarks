import setuptools, sys

setup_args = dict(
    author="deathbeds",
    long_description="description",
    long_description_content_type='text/markdown',
    python_requires=">=3.5",
    # license="BSD-3-Clause",
    setup_requires=[
        'pytest-runner',
        'twine>=1.11.0',
        'setuptools>=38.6.',
    ] + ([] if sys.version_info.minor == 4 else ['wheel>=0.31.0']),
    tests_require=['pytest'],
    install_requires=[],
    packages=[],
    zip_safe=False,
)

if __name__ == "__main__":
    setuptools.setup(**setup_args)
