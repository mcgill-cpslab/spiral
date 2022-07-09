# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/mcgill-cpslab/spiral/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

spirograph could always use more documentation, whether as part of the
official spirograph docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/mcgill-cpslab/spiral/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `spirograph` for local development.

1. Find an issue you want to work on..
2. Clone `spirograph` locally

    ```
    $ git clone https://github.com/mcgill-cpslab/spiral.git
    ```

3. Ensure [poetry](https://python-poetry.org/docs/) is installed.
4. Install dependencies and start your virtualenv:

    ```
    $ poetry install -E test -E doc -E dev
    ```

5. Create your local development branch.
   You can create your local development branch from the project branch with name prefix as 'project/' for local development:

    ```
    $ git checkout project/a_project_branch
    $ git checkout -b feature/name-of-your-feature
    ```
    
    for bug fix:

    ```
    $ git checkout -b bugfix/name-of-your-bugfix
    ```

    Now you can make your changes locally.

    If you have spent quite some time on your change and there maybe new updates to the master branch, 
    run the following to sync up your feature branch:


    ```
    $ git checkout project/base_project_branch
    $ git pull
    $ git checkout your_feature_bugfix_branch
    $ git merge --no-ff  project/base_project_branch
    ```


6. When you're done making changes, check that your changes pass the
   tests, including testing other Python versions, with tox:

    ```
    $ poetry run tox
    ```

7. Commit your changes and push your branch to GitHub:

    ```
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-branch
    ```

8. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for Python 3.6, 3.7, 3.8 and 3.9. Check
   https://github.com/mcgill-cpslab/spiral/actions
   and make sure that the tests pass for all supported Python versions.

## Tips

```
$ poetry run pytest tests/your_working_module/your_test_cases.py
```

To run a subset of tests.


```
$ poetry run flake8 pyth_to_your_code
```

To run coding style check on a particular file or folder

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in CHANGELOG.md).
Then run:

```
$ poetry run bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags
```

GitHub Actions will then deploy to PyPI if tests pass.
