# Release docs

## PyPI release

1. Update version number in `pyproject.toml` and `__init__.py` files.
2. Run `uv lock`
3. Run `make test` & `make pre-commit`
4. Commit, tag and push changes.
    ```shell
    git add .
    git commit -m "chore: bump version to x.y.z"
    git push origin  # wait for all CI jobs to succeed 
    git tag x.y.z
    git push origin --tags 
    ```
5. Go to [GitHub in the tags section](https://github.com/mlco2/ecologits/tags), on the latest tag an click "Create release".
6. Click on "Generate release notes" and review the changelog.
7. Click "Publish release".
8. Go to [GitHub Actions](https://github.com/mlco2/ecologits/actions), check that the CI release job succeed.


## Publish versioned docs without a PyPI release

Use this rarely, when docs or blog content must be published under the current version and `latest`, without creating a GitHub release, PyPI release, or git tag.

Use the MAJOR.MINOR version number (e.g. `0.11` and not `0.11.0`). 

Do not force-push `gh-pages`.

1. Commit and push the docs changes to `main`.
2. Wait for the `publish-docs` workflow to publish `dev`.
3. Publish the same docs as the versioned `latest` docs.
    ```shell
    git fetch origin gh-pages
    git branch -f gh-pages origin/gh-pages
    uv run mike deploy -b gh-pages <MAJOR.MINOR> latest --update-aliases --push
    git fetch origin gh-pages
    git show origin/gh-pages:versions.json
    ```
