# Contributing to RTC-Tools 2

There are many ways you can contribute to RTC-Tools, such as:

- **Reporting issues**: If you encounter any bugs, errors, or unexpected behavior while using RTC-Tools, 
please report them on our [issue tracker](https://github.com/rtc-tools/rtc-tools/issues).
Please follow the issue template and provide as much information as possible to help us reproduce and fix the issue.
- **Suggesting features**: If you have any ideas or suggestions for new features or improvements, 
please share them on our [issue tracker](https://github.com/rtc-tools/rtc-tools/issues).
Please use the appropriate category and tag for your topic and explain your motivation and use case clearly.
- **Submitting merge requests**: If you want to contribute code or documentation to RTC-Tools, 
please fork the repository and create a merge request.
Please write docstrings for your functions and classes and make sure your changes pass the tests and checks before submitting.
Please also add a brief description of your changes and reference any related issues or discussions.
- **Improving documentation**: If you find any errors, typos, or inconsistencies in the documentation,
 or if you want to add more examples, tutorials, or explanations, 
 please feel free to edit the documentation files in the docs folder and submit a merge request.
Please follow the [Sphinx syntax and style guide](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) for writing documentation.
- **Reviewing merge requests**: If you are familiar with RTC-Tools and want to help us maintain the quality and consistency of the code and documentation,
 please review the open merge requests and provide constructive feedback and suggestions.
- **Testing new releases**: If you want to help us test the new features and bug fixes before they are officially released, 
please install the latest development version of RTC-Tools from the [GitHub repository](https://github.com/rtc-tools/rtc-tools)
 and report any issues or feedback on the [issue tracker](https://github.com/rtc-tools/rtc-tools/issues).

## Guidelines for creating issues
1. **Title**: Provide a concise and informative title. The title should summarize the problem.
2. **Description**: Describe the issue in detail. Include steps to reproduce the issue, expected behavior, and actual behavior.
3. **Labels**: Use labels to categorize the issue. This helps in prioritizing and resolving issues.
4. **Screenshots**: If applicable, add screenshots to help explain the problem.
5. **Environment**: Mention the version of RTC-Tools, Python and external packages you're using (CasADi, Pymoca, numpy), along with relevant details about your operating system.


## Guidelines for creating merge requests
1. **Identify or create issue**: Before making any changes,
please open an issue following the guidelines above,
or comment on an existing one in the
[issue tracker](https://github.com/rtc-tools/rtc-tools/issues)
to discuss your ideas with the maintainers.
This will help us avoid duplication of work
and ensure that your contribution aligns with the goals of the project.
2. **Branch**: Create a new branch for each merge request. The branch name should be descriptive and reflect the changes being made.
3. **Commit**: Please make your commits following the guidelines in the [Commits and Commit Messages](#commits-and-commit-messages) section below.
4. **Write tests**: If possible, write tests that cover your changes, and add them to the `tests` folder.
This will help ensure that your changes result in the desired behavior and that functionalities will not accidently break in the future.
5. **Documentation**: Please also update the documentation if necessary and add examples to the `examples` folder.
6. **Create merge request**: Mention the corresponding issue from the [issue tracker](https://github.com/rtc-tools/rtc-tools/issues).
Describe what changes you've made, why you've made them, and how they address the issue at hand.
7. **Code review**: Request code review from your peers. Address any comments or suggestions they might have.


## Commits and Commit Messages

Each commit ideally satisfies the following:

- Each commit has a clear and single purpose.
- After each commit, all unit tests should still pass.

Commit messages should have the following structure:

```text
<scope>: <short description>

<complete description>
```

- scope: explains which part of the code is affected, e.g.:
    - optimization (only affects the optimization part)
    - homotopy_mixin (only affects the homotopy_mixin module)
    - tests (only affects the tests)
    - doc (only affects the documentation)
- short description: describes what is changed in the commit with a single sentence.
- complete description: explain in detail what is done in the commit and why.
    This can take up multiple paragraphs.


## Setting up a development environment

To set up your development environment, you will need:

- Python 3.10 or higher (up to 3.14)
- Git

You can clone the repository and install it from source:

```bash
git clone https://github.com/rtc-tools/rtc-tools.git
cd rtc-tools
uv sync
```

To ensure that your code meets our standards, we recommend using pre-commit.
Run the following command to set up the pre-commit hook:

```bash
pre-commit install
```

This will automatically check your code for formatting and linting issues before each commit.


To run the tests:

```bash
pytest tests
```

To build the documentation, the required dependencies are in the `docs` dependency group from `pyproject.toml`.
Run these commands from the repository root:

```bash
uv sync --group docs
uv run sphinx-build -b html doc doc/_build/html
```

The built HTML pages will be in `doc/_build/html/`.

## Version numbering and release cycle

For version numbers we use the guidelines described in <https://semver.org>:

> Given a version number MAJOR.MINOR.PATCH, increment the:
> 
> 1. MAJOR version when you make incompatible API changes
> 2. MINOR version when you add functionality in a backward compatible manner
> 3. PATCH version when you make backward compatible bug fixes
> 
> Additional labels for pre-release and build metadata are available
> as extensions to the MAJOR.MINOR.PATCH format.

The development of a new MINOR version in RTC-Tools consists of four stages:

1. Alpha (a): Version that is not yet feature complete and may contain bugs.
    Each alpha release can either fix bugs or add new features.
2. Beta (b): Version that is feature complete but is likely to contain bugs.
    After a beta version has been created, no new features can be added anymore.
    A beta version is tested more thoroughly.
3. Release candidate (rc): Version that has been tested through the beta versions releases
    and can now be tested as if it were the stable release.
    If bugs still pop up, new RC versions can be created to fix them.
    Additions are allowed but should be code-unrelated,
    such as changes to the documentation required for the release.
4. Stable release: Final version that has passed all tests.

There can be multiple alpha-, beta-, and rc-versions,
but we should not go back to a previous stage.

An example of a release sequence is:

- 2.6.0a1 Add a feature.
- 2.6.0a2 Add another feature and fix a bug.
- 2.6.0b1 First beta release.
- 2.6.0b2 Fixed a bug.
- 2.6.0b3 Fixed another bug.
- 2.6.0rc1 First release candidate after having tested thoroughly.
- 2.6.0rc2 Fixed a bug that did not show up in the standard tests.
- 2.6.0 **Stable release**.
    No changes were made after last release candidate.
- 2.6.1 Fixed a bug.
- 2.6.2 Fixed another bug.

If we start with a new release cycle for X.Y+1,
and still want to fix a bug for the previous version X.Y,
we create a separate branch `maintenance/X.Y` where we add patches for X.Y.

## Release Notes

Before creating a release, make sure that the release notes are updated in
[RELEASE_NOTES.md](RELEASE_NOTES.md).


## Contact

If you have any questions or comments about RTC-Tools, please contact us at rtctools@deltares.nl.

We hope you enjoy using and contributing to RTC-Tools!