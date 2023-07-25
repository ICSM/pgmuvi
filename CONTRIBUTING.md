# Contributing to `pgmuvi`

## Code of Conduct

We expect all contributors to conduct themselves in a professional and respectful
manner. Please be kind and courteous to others, and do not engage in any behavior
that would be considered harassment of any kind. Bullying, discrimination, insulting, 
or other disrespectful behavior will not be tolerated. Do not make any comments that
others might reasonably find offensive. Disagreements are fine, but please be polite
and respectful. If you are asked to stop a particular behavior, please do so
immediately. Failure to follow this code of conduct may result in your removal from
the project and/or other consequences. If you believe that someone has violated
this code of conduct, please contact the project maintainers. This code of conduct 
extends both to interactions on the project's github page and to interactions in
other project spaces, including but not limited to email, Slack, and in-person
meetings. 

## Getting Started

We welcome contributions from anyone, regardless of experience level - we are
happy to help you get started, and we were all new once! If you are new to open
source, we recommend reading [this guide](https://opensource.guide/how-to-contribute/).
If you are new to git, we recommend reading [this guide](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics).

If you are new to `pgmuvi`, we recommend starting by reading the [documentation](https://pgmuvi.readthedocs.io/en/latest/).



## How to Contribute

There are many ways to contribute to `pgmuvi`. You can ask questions, report
issues, contribute code, improve documentation, or help with testing. We welcome
all contributions, large or small. Please don't feel like you need to be an
expert to contribute! 

## Asking Questions

If you have a question about how to use `pgmuvi`, please open an issue on github
and tag it with the `question` label. Please provide as much detail as possible
about what you are trying to do and what you have tried so far. If you are
getting an error message, please include the full error message, including the
stack trace if there was one. If you are trying to do something that you think
should be possible but isn't, please explain why you think it should be possible
and what you have tried so far.

## Reporting Issues

If you believe you've found an issue, please report it along with a detailed
explanation of what you were trying to do and what went wrong. Please include
the following information:

- The version of `pgmuvi` you are using
- The operating system you are using, python version, and the versions of any
  dependencies (e.g. torch, numpy, etc.). It particularly helps if you can
    provide a `pip freeze` output (or equivalent).
- A minimal example that reproduces the issue. Ideally, this should be a code snippet 
    or a python script that can be run from the command line. If you can't provide
    this, please provide a detailed description of what you were trying to do and
    what went wrong.
- The full error message you received, including the stack trace if there was one.

## Contributing Code

We sincerely welcome contributions to `pgmuvi`. If you would like to contribute
code, please open an issue first to discuss your ideas with the developer community, 
or respond to an open issue indicating that you are interested in working on it. 
This helps us avoid duplicate work and ensures that your contribution is likely to
be accepted. 

Once you have an issue number, please fork the repository and create a new branch
whose name clearly identifies the feature or bug you are working on (e.g. `issue-1234`,
`feature-foo`, `bugfix-bar`, etc.). When you are ready to submit your code, please
open a pull request and reference the issue number in the description. 

We will review your pull request as soon as possible. If we request changes, please
address them in a new commit and push to the same branch, or provide a brief explanation 
of why you think our suggestion won't work. Once we are satisfied with
your changes, we will merge your pull request. 

## Code Style

We try to follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
as much as possible. We lint our code using [ruff](https://beta.ruff.rs/docs/), and we recommend that you do
the same. There is a github action that will run `ruff` on your code when you open
a pull request, and we will ask you to address any issues that it finds.

## Testing

Please ensure that your code includes tests that cover any new functionality you
have added. If you are fixing a bug, please include a test that would have caught
the bug. We like to have both unit tests and integration tests. 

If you think you've identified a gap in our tests (with our current tests that 
probably isn't too hard!), we welcome contributions to our test suite. If you are 
not sure how to write tests, please open an issue and we will be happy to help 
you get started. New tests *do not* need to be associated new features or bug 
fixes - we welcome tests that improve our coverage of existing functionality.


## Documentation

Please ensure that your code includes documentation that explains what it does
and how to use it. We use [sphinx](https://www.sphinx-doc.org/en/master/) to
generate our documentation. Please make sure that *all* new objects, modules,
functions, classes, and methods have docstrings following the 
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format.

If you just enjoy writing documentation, we welcome contributions to our
documentation that are not attached to any particular code change. If you are
not sure how to get started, please open an issue and we will be happy to help. 
Tutorials and examples are particularly welcome.
