# Security Policy

## Supported Versions

scFair is under active development. Security fixes are applied to the latest
released version on [PyPI](https://pypi.org/project/scfair/); we do not maintain
long-term security support for older releases.

| Version | Supported |
| ------- | --------- |
| Latest release | :white_check_mark: |
| Older releases | :x: |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, use GitHub's private vulnerability reporting:
[Report a vulnerability](https://github.com/leelieber2025/scFair/security/advisories/new)
(repository → **Security** tab → **Report a vulnerability**).

Include a description of the issue, steps to reproduce, and the affected
version. We will acknowledge reports and follow up with a timeline for a fix
once the report is triaged.

## Scope

scFair is a local single-cell analysis library (no network services, no
telemetry, no remote code execution by design). Relevant concerns include
unsafe handling of untrusted input files and path traversal in optional
on-disk snapshot paths.
