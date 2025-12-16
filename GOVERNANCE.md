# RTC-Tools Governance

## Table of Contents

- [Project Governance](#project-governance)
  - [Governance Roles](#governance-roles)
  - [Operational Roles](#operational-roles)
  - [TSC Voting](#tsc-voting)
  - [Becoming a Committer](#becoming-a-committer)
  - [Open Participation](#open-participation)
  - [Code of Conduct](#code-of-conduct)
- [Release Management](#release-management)
  - [Version Numbering](#version-numbering)
  - [Release Cycle](#release-cycle)
  - [Branch Management](#branch-management)
- [Documentation Standards](#documentation-standards)
- [Merging Policy and Backwards Compatibility](#merging-policy-and-backwards-compatibility)
  - [Review and Approval Process](#review-and-approval-process)
  - [Linear History](#linear-history)
  - [Backwards Compatibility](#backwards-compatibility)

## Project Governance

RTC-Tools is governed by a Technical Steering Committee (TSC), responsible for technical oversight, community norms, workflows, and project direction. The TSC operates according to the rules defined in the [Technical Charter](CHARTER.md).  

### Governance Roles

- **Contributors**: Anyone who contributes code, documentation, or other artifacts.
- **Committers**: Contributors who have earned the ability to modify ("commit") source code, documentation, or other technical artifacts.
- **TSC Members**: Initially, the Project's Committers. The TSC may update its membership and roles as documented here.

### Operational Roles

Contributors, Committers, and TSC Members can assume different operational roles, depending on their expertise, interests, and availability. These roles help distribute project responsibilities and ensure smooth operations:

- **Release Manager**: Coordinates releases, manages version numbering, ensures release quality, and oversees release processes.
- **Maintainer**: Day-to-day code maintenance, pull request reviews, and issue triage.
- **Community Manager**: Facilitates community engagement, manages communications, supports contributor onboarding, and moderates discussions.
- **Security, Quality and DevOps Lead**: Handles security vulnerabilities, maintains testing standards, ensures code quality processes, oversees CI/CD infrastructure, and handles DevOps operations.
- **Documentation Lead**: Maintains documentation quality, coordinates documentation efforts, and ensures documentation remains up-to-date with software changes.

### TSC Voting

The TSC operates openly and transparently. Meetings are open to the public and may be held electronically or in person.  
Decisions are made by consensus when possible; if a vote is required, each voting member has one vote. 
- Quorum for TSC meetings is at least 50% of voting members.
- Decisions at meetings require a majority of those present, provided quorum is met.
- Electronic votes require a majority of all voting members.
- If a vote cannot be resolved, any voting member may refer the matter to the Series Manager for assistance, as specified in the [Technical Charter](CHARTER.md#3-tsc-voting).

### Becoming a Committer

Anyone can contribute to RTC-Tools by following the [Guidelines for Creating Merge Requests](CONTRIBUTING.md#guidelines-for-creating-merge-requests). Contributors who have demonstrated sustained commitment to the project may be nominated to become Committers. The path typically involves:

1. **Building a track record**: Make regular, high-quality contributions as described in [Ways to Contribute](CONTRIBUTING.md#ways-to-contribute).
2. **Nomination**: Any existing Committer can nominate a Contributor, or Contributors can express interest by contacting an existing Committer or emailing [info@rtctools.energy](mailto:info@rtctools.energy).
3. **Review and approval**: The nomination is discussed among existing Committers and must be approved by a majority vote.

Committers may be removed by a majority vote of the other existing Committers, following the same voting procedures described in [TSC Voting](#tsc-voting).

### Open Participation

Participation is open to anyone, regardless of affiliation, as long as they abide by the [Technical Charter](CHARTER.md) and the project's policies.

### Code of Conduct

RTC-Tools follows the [LF Projects Code of Conduct](https://lfprojects.org/policies/).  
All participants are expected to act respectfully and professionally.

For more details, see the [Technical Charter](CHARTER.md).

## Release Management

### Version Numbering

For version numbers we use the guidelines described in <https://semver.org>:

> Given a version number MAJOR.MINOR.PATCH, increment the:
>
> 1. MAJOR version when you make incompatible API changes
> 2. MINOR version when you add functionality in a backward compatible manner
> 3. PATCH version when you make backward compatible bug fixes
>
> Additional labels for pre-release and build metadata are available
> as extensions to the MAJOR.MINOR.PATCH format.

### Release Cycle

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
4. Stable release: Final version that has passed all tests. See [Support Policy](#support-policy) for information on how long stable versions are supported and what support users can expect.

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

### Branch Management

If we start with a new release cycle for X.Y+1,
and still want to fix a bug for the previous version X.Y,
we create a separate branch `maintenance/X.Y` where we add patches for X.Y.

Bug fixes for previous stable versions should be submitted to the corresponding `maintenance/X.Y` branch.

### Support Policy

- **Active Support**: The latest stable MINOR version (X.Y.0) receives bug fixes via PATCH releases (X.Y.Z). Active support continues until the next MINOR version is released.

- **Security Fixes**: Critical security vulnerabilities in the current MINOR version are addressed promptly. Previous MINOR versions may receive security patches at the TSC's discretion.

- **Maintenance Branches**: Bug fixes for previous stable versions are maintained in `maintenance/X.Y` branches as described in [Branch Management](#branch-management).

- **End-of-Life**: When a new MINOR version is released, the previous version enters maintenance mode. The TSC determines the end-of-life timeline based on community needs and resource availability.

## Documentation Standards

Documentation is a critical part of the RTC-Tools project. All contributions should adhere to these standards:

- **API Documentation**: All public APIs must be documented using docstrings following [PEP 257](https://peps.python.org/pep-0257/) conventions.
- **User Documentation**: User-facing documentation should be written in clear, accessible language and maintained in the `docs` directory using Sphinx.
- **Examples**: Code examples should be provided for key functionality and kept up-to-date with the current API.
- **Breaking Changes**: All breaking changes must include documentation updates explaining the changes and migration paths.
- **Release Notes**: Each release must include comprehensive release notes detailing new features, bug fixes, and any breaking changes.

The [Documentation Lead](#operational-roles) is responsible for ensuring documentation quality and consistency across the project.

## Merging Policy

All contributions to RTC-Tools are reviewed and merged according to the following policy, in line with the project's [Technical Charter](CHARTER.md):

### Review and Approval Process

- **Review Requirements**: All pull requests must be reviewed by at least one maintainer or committer before being merged. The Technical Steering Committee (TSC) may require additional reviews for significant or controversial changes.
- **Consensus and Voting**: The project aims to operate by consensus. If consensus cannot be reached, the TSC may call a vote as described in the [Technical Charter](CHARTER.md).
- **Release Approval**: The TSC approves releases containing MAJOR version changes or backwards-incompatible changes before the first release candidate (rc1). The Release Manager coordinates approval following [TSC Voting](#tsc-voting) procedures. MINOR and PATCH releases may proceed without TSC approval unless backwards-incompatible.

### Linear History

We maintain a linear Git history by using git rebase instead of git merge. This approach provides several benefits:
- Cleaner, more readable project history
- Easier bisecting and debugging
- Reduced complexity in the commit graph
- Makes it straightforward to review and edit commit messages

The GitHub repository is configured to automatically rebase pull requests when merging to master. Contributors should rebase their branches on the latest main branch before submitting pull requests and use `git commit --amend` or `git rebase -i` to keep commits clean.

**Resources for learning Git rebase and amend:**
- [Git Documentation: git-rebase](https://git-scm.com/docs/git-rebase)
- [Atlassian Git Tutorial: Rewriting History](https://www.atlassian.com/git/tutorials/rewriting-history)
- [GitHub Docs: About Git rebase](https://docs.github.com/en/get-started/using-git/about-git-rebase)

### Backwards Compatibility

- CI tests are run to ensure that changes do not break existing functionality and maintain API compatibility. Deltares maintains an additional test bench in an internal TeamCity environment with test models that are not open to the public. Though it is not publically available, this will be used to report regressions and possible bugs back upstream at the earliest possibility.
- Changes that break backwards compatibility (i.e., incompatible API changes) should only be made when necessary and must be clearly documented, following our [Documentation Standards](#documentation-standards).
- Contributors are encouraged to maintain backwards compatibility whenever possible. Deprecation warnings should be provided before removing or changing public APIs.

By following this merging policy, RTC-Tools ensures a stable and predictable experience for all users and contributors, in accordance with the project's [Technical Charter](CHARTER.md).

#### Edge Cases: What Constitutes a Breaking Change

To clarify common edge cases specific to RTC-Tools, the following are **breaking changes** requiring a MAJOR version increment:

- Changing class variable defaults (e.g., `csv_delimiter` from `","` to `";"`, or `csv_ensemble_mode` from `False` to `True`) if existing user models rely on the current default behavior
- Changing method signatures that users override (e.g., adding a required parameter to `objective()`, removing a parameter, or changing parameter types)
- Changing expected data formats (e.g., renaming columns in CSV files, changing Modelica variable naming conventions, or modifying file structure requirements)
- Removing or renaming public class variables, methods, or attributes that users may access

The following are **not breaking changes** and require only a MINOR version increment:

- Adding optional method parameters with sensible defaults that maintain backwards compatibility
- Adding new class variables with defaults that preserve existing behavior
- Adding new public methods or attributes that don't affect existing functionality

When in doubt, treat a change as breaking if it could reasonably cause existing user code or models to fail or behave differently.
