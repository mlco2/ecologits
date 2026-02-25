# Release

1. Update version number in `pyproject.toml` and `__init__.py` files.
2. Run `uv lock`
3. Commit, tag and push changes.
    ```shell
    git add .
    git commit -m "chore: bump version to x.y.z"
    git push origin  # wait for all CI jobs to succeed 
    git tag x.y.z
    git push origin --tags 
    ```
4. Go to [GitHub in the tags section](https://github.com/mlco2/ecologits/tags), on the latest tag an click "Create release".
5. Click on "Generate release notes" and review the changelog.
6. Click "Publish release".
7. Go to [GitHub Actions](https://github.com/mlco2/ecologits/actions), check that the CI release job succeed.
