# Contributing to RTC-Tools 2

## Table of Contents

- [Initial Committers](#initial-committers)
- [Ways to Contribute](#ways-to-contribute)
- [Guidelines for Creating Issues](#guidelines-for-creating-issues)
  - [Security Reporting](#security-reporting)
- [Guidelines for Creating Merge Requests](#guidelines-for-creating-merge-requests)
- [Commits and Commit Messages](#commits-and-commit-messages)
- [Code Quality Guidelines](#code-quality-guidelines)
- [Setting up a Development Environment](#setting-up-a-development-environment)
- [Project Governance](#project-governance)
- [Licensing and Developer Certificate of Origin](#licensing-and-developer-certificate-of-origin-dco)
- [Contact](#contact)

## Initial Committers

The initial committers of the RTC-Tools project are:

- Jorn Baayen
- Ailbhe Mitchell
- Tjerk Vreeken
- Teresa Piovesan
- Bernhard Becker
- Farid Alavi
- Joris Gillis
- Klaudia Horváth
- Jesús A. Rodríguez-Sarasty

## Ways to Contribute

There are many ways you can contribute to RTC-Tools, such as:

- **Reporting issues**: If you encounter any bugs, errors, or unexpected behavior while using RTC-Tools, 
please report them on our [issue tracker](https://github.com/rtc-tools/rtc-tools/issues).
Please follow the issue template and provide as much information as possible to help us reproduce and fix the issue.
- **Suggesting features**: If you have any ideas or suggestions for new features or improvements, 
please share them on our [issue tracker](https://github.com/rtc-tools/rtc-tools/issues).
Please use the appropriate category and tag for your topic and explain your motivation and use case clearly.
- **Submitting pull requests**: If you want to contribute code or documentation to RTC-Tools, please create a pull request. Before submitting, please follow the [Guidelines for Creating Merge Requests](#guidelines-for-creating-merge-requests) below.  
- **Improving documentation**: If you find any errors, typos, or inconsistencies in the documentation,
 or if you want to add more examples, tutorials, or explanations, 
 please feel free to edit the documentation files in the docs folder and submit a merge request.
Please follow the [Sphinx syntax and style guide](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) for writing documentation.
- **Reviewing merge requests**: If you are familiar with RTC-Tools and want to help us maintain the quality and consistency of the code and documentation,
 please review the open merge requests and provide constructive feedback and suggestions.
- **Testing new releases**: If you want to help us test the new features and bug fixes before they are officially released, 
please install the latest development version of RTC-Tools from the [GitHub repository](https://github.com/rtc-tools/rtc-tools)
 and report any issues or feedback on the [issue tracker](https://github.com/rtc-tools/rtc-tools/issues).

## Guidelines for Creating Issues

1. **Title**: Provide a concise and informative title. The title should summarize the problem.
2. **Description**: Describe the issue in detail. For bug reports, include steps to reproduce the issue, expected behavior, and actual behavior. Also mention the versions of RTC-Tools, Python and external packages you're using (CasADi, Pymoca, numpy), along with relevant details about your operating system.
3. **Minimal reproducible example**: Whenever possible, include a minimal reproducible example that demonstrates the issue.
    - The example should use the smallest amount of code and data necessary to reproduce the problem.
    - If you cannot create a minimal example, simplify your model and remove unnecessary data as much as possible while still reproducing the issue.
    - Safe data formats that can be shared publicly include CSV, JSON, or small XML files.
    - If you do not want to make your data publicly available in the issue tracker, please report your issue by emailing [info@rtctools.energy](mailto:info@rtctools.energy).
4. **Labels**: Use labels to categorize the issue. This helps in prioritizing and resolving issues.

### Security Reporting

If you discover a security vulnerability, please report it responsibly by emailing [info@rtctools.energy](mailto:info@rtctools.energy) rather than opening a public issue. We will assess the vulnerability and keep you updated on progress and next steps.

**What qualifies as a security vulnerability?**
- Compromised or malicious dependencies
- Code execution vulnerabilities or insufficient input sanitization
- Exposure of sensitive data or credentials
- Input data (known or crafted) that causes crashes or data corruption

## Guidelines for Creating Merge Requests

1. **Identify or create an issue**: Before making any changes, open an issue following the [guidelines](#guidelines-for-creating-issues) above, or comment on an existing one in the [issue tracker](https://github.com/deltares/rtc-tools/issues) to discuss your ideas with the maintainers. This helps avoid duplication and ensures your contribution aligns with project goals and the [governance model](GOVERNANCE.md).
2. **Fork or Branch**:
    - New Contributors: Fork the repository and create a new branch in your fork.
    - Committers: Create a new branch directly in the main repository. Use a descriptive branch name, such as `feat/short-description`, `fix/issue-123`, or `docs/update-readme`.
3. **Commit**: Make clear, focused commits following the [Commits and Commit Messages](#commits-and-commit-messages) guidelines.
4. **Write tests**: If possible, write tests that cover your changes and add them to the `tests` folder. This helps ensure your changes work as intended and prevent regressions. For tests requiring data:
    - Use existing test datasets from the `tests/data` directory when applicable.
    - For new functionality requiring specific test data, include minimal test datasets with your contribution. Keep test data files small and focused.
5. **Documentation**: Update documentation and add examples to the `examples` folder if necessary.
6. **Create a pull request**: Reference the corresponding issue in your PR description. Clearly describe what changes you've made, why, and any relevant context.
7. **Check CI status**: Ensure all automated checks and tests pass before requesting a review.
8. **Request review**: Request a code review from a maintainer or committer (see [Governance Roles](GOVERNANCE.md#governance-roles) and [Operational Roles](GOVERNANCE.md#operational-roles)) and address any comments or suggestions.

Contributors should rebase their branches on the latest main branch before submitting pull requests. As described in our [Governance](GOVERNANCE.md#linear-history) document, we maintain a linear Git history instead of using merge commits. To keep the history clean, use `git commit --amend` or `git rebase -i` to amend commits for small changes (typos, linting, formatting) rather than creating separate fixup commits.

Keep pull requests small and focused for easier review and faster merging. A focused PR addresses a single concern (e.g., one bug fix, one feature, or one refactoring). While there's no strict limit on lines of code or number of commits, both the overall changeset and individual commits should be reviewable. Each commit should follow the [Commits and Commit Messages](#commits-and-commit-messages) guidelines.

## Commits and Commit Messages

Each commit ideally satisfies the following:

- Each commit has a clear and single purpose.
- After each commit, all unit tests should still pass.

We recommend using the [Conventional Commits](https://www.conventionalcommits.org/) format for commit messages. This enables automatic changelog generation and better tooling support.

Commit messages should have the following structure:

```text
<type>(<scope>): <short description>

<complete description>
```

- type: the kind of change, e.g.:
    - `feat`: new feature
    - `fix`: bug fix
    - `docs`: documentation changes
    - `test`: adding or updating tests
    - `refactor`: code refactoring
    - `perf`: performance improvements
    - `chore`: maintenance tasks
- scope: which part of the code is affected, e.g.:
    - `optimization`: only affects the optimization part
    - `homotopy_mixin`: only affects the homotopy_mixin module
- short description: describes what is changed in the commit with a single sentence.
- complete description: explain in detail what is done in the commit and why.
    This can take up multiple paragraphs.


## Code Quality Guidelines

To maintain a high standard of code quality in RTC-Tools, please follow these guidelines when contributing:

- **Type Annotations**: Use [PEP 484](https://peps.python.org/pep-0484/) type hints where appropriate to improve code clarity and enable static analysis. For union types, prefer the modern syntax available in Python 3.10+:
  ```python
  x: str | None
  ```
  instead of:
  ```python
  from typing import Optional
  x: Optional[str]
  ```
- **Docstrings**: Add clear and concise docstrings to all public modules, classes, functions, and methods. Use [PEP 257](https://peps.python.org/pep-0257/) conventions.
- **Pre-commit Hooks**: Use the provided pre-commit configuration by running `pre-commit install` to automatically check formatting and linting before each commit.
- **Follow PEP 8**: Write Python code that adheres to [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- **Code coverage**: Aim for good test coverage to ensure code quality and prevent regressions.
- **Avoid Code Duplication**: Reuse existing utilities and functions where possible. Refactor code to eliminate duplication.
- **Readability**: Write code that is easy to read and understand. Use meaningful variable and function names.
- **Error Handling**: Handle exceptions gracefully and provide informative error messages.
- **Dependencies**: Only add new dependencies if necessary and discuss them with maintainers first.
- **Performance**: Consider the performance impact of your changes, especially in core computation routines.

By following these guidelines, you help ensure RTC-Tools remains robust, maintainable, and accessible.

## Setting up a Development Environment

To set up your development environment, you will need:

- Python 3.10 or higher (up to 3.14)
- Git

You can clone the repository and install it from source:

```bash
git clone https://github.com/rtc-tools/rtc-tools.git
cd rtc-tools
uv sync
```

This installs the package along with all development dependencies (testing, linting, documentation tools).

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

To build the documentation:

```bash
cd doc
make html
```

## Project Governance

RTC-Tools is governed according to the [Technical Charter](CHARTER.md) and detailed in the [Governance](GOVERNANCE.md) document, which together establish:

- The project structure and roles
- Decision-making processes
- Contribution guidelines
- Code of conduct requirements

All contributors are expected to follow the governance model outlined in the [Technical Charter](CHARTER.md) and [Governance](GOVERNANCE.md) documents.

## Licensing and Developer Certificate of Origin (DCO)

- **Code contributions**: All code must be contributed under the GNU Lesser General Public License v3.0 (LGPL-3.0).
  See [COPYING](https://github.com/Deltares/rtc-tools/blob/master/COPYING) and [COPYING.LESSER](https://github.com/Deltares/rtc-tools/blob/master/COPYING.LESSER) for details.
- **Documentation contributions**: All documentation is licensed under the [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
- **SPDX Identifiers**: Please include appropriate SPDX license identifiers in new files.

All contributions must also be signed off using the [Developer Certificate of Origin (DCO)](https://developercertificate.org/):

```text
Signed-off-by: Your Name <your.email@example.com>
```

You can add this automatically with:

```bash
git commit -s
```

## Contact

If you have any questions or comments about RTC-Tools, please contact us at info@rtctools.energy.

We hope you enjoy using and contributing to RTC-Tools!