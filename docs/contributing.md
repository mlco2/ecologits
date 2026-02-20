# Contributing

Help us improve EcoLogits by contributing! :tada:

## Issues

Questions, feature requests and bug reports are all welcome as [discussions or issues](https://github.com/mlco2/ecologits/issues/new/choose).

When submitting a feature request or bug report, please provide as much detail as possible. For bug reports, please include relevant information about your environment, including the version of EcoLogits and other Python dependencies used in your project.

## Pull Requests

Getting started and creating a Pull Request is a straightforward process. Since EcoLogits is regularly updated, you can expect to see your contributions incorporated into the project within a matter of days or weeks.

For non-trivial changes, please create an issue to discuss your proposal before submitting pull request. This ensures we can review and refine your idea before implementation.

### Prerequisites

You'll need to meet the following requirements:

- **Python version above 3.9**
- **git**
- **make**
- **[poetry](https://python-poetry.org/docs/#installation)**
- **[pre-commit](https://pre-commit.com/#install)**

### Installation and setup

Fork the repository on GitHub and clone your fork locally.

```shell
# Clone your fork and cd into the repo directory
git clone git@github.com:<your username>/ecologits.git
cd ecologits

# Install ecologits development dependencies with poetry
make install
```

### Check out a new branch and make your changes

Create a new branch for your changes.

```shell
# Checkout a new branch and make your changes
git checkout -b my-new-feature-branch
# Make your changes and implements tests...
```

### Run tests

Run tests locally to make sure everything is working as expected.

```shell
make test
```

If you have added a new provider you will need to record your tests with VCR.py through [pytest-recording](https://github.com/kiwicom/pytest-recording).

```shell
make test-record
```

Once your tests are recorded, please check that the newly created cassette files (located in `tests/cassettes/...`) do not contain any sensible information like API tokens. If so you will need to update the configuration accordingly in `conftest.py` and run again the command to record tests.

### Build documentation

If you've made any changes to the documentation (including changes to function signatures, class definitions, or docstrings that will appear in the API documentation), make sure it builds successfully.

```shell
# Build documentation
make docs
# If you have changed the documentation, make sure it builds successfully.
```

You can also serve the documentation locally.

```shell
# Serve the documentation at localhost:8000
poetry run mkdocs serve
```

### Code formatting and pre-commit

Before pushing your work, run the pre-commit hook that will check and lint your code.

```shell
# Run all checks before commit
make pre-commit
```

### Commit and push your changes

Commit your changes, push your branch to GitHub, and create a pull request.

Please follow the pull request template and fill in as much information as possible. Link to any relevant issues and include a description of your changes.

When your pull request is ready for review, add a comment with the message "please review" and we'll take a look as soon as we can.

## Documentation style

Documentation is written in Markdown and built using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). API documentation is build from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

### Code documentation

When contributing to EcoLogits, please make sure that all code is well documented. The following should be documented using properly formatted docstrings.

We use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) formatted according to [PEP 257](https://peps.python.org/pep-0257/) guidelines. (See [Example Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for further examples.)

### Documentation style

Documentation should be written in a clear, concise, and approachable tone, making it easy for readers to understand and follow along. Aim for brevity while still providing complete information.

Code examples are highly encouraged, but should be kept short, simple and self-contained. Ensure that each example is complete, runnable, and can be easily executed by readers.

## Acknowledgment

We'd like to acknowledge that this contribution guide is heavily inspired by the excellent [guide from Pydantic](https://docs.pydantic.dev/latest/contributing/). Thanks for the inspiration! :heart:
