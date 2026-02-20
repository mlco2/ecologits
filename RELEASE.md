# Release

1. Update version number in `pyproject.toml` and `__init__.py` files.
2. Commit, tag and push changes.
    ```shell
    git add .
    git commit -m "chore: bump version to x.y.z"
    git push origin  # wait for all CI jobs to succeed 
    git tag x.y.z
    git push origin --tags 
    ```
3. Go to [GitHub in the tags section](https://github.com/mlco2/ecologits/tags), on the latest tag an click "Create release".
4. Click on "Generate release notes" and review the changelog.
5. Click "Publish release".
6. Go to [GitHub Actions](https://github.com/mlco2/ecologits/actions), check that the CI release job succeed.
