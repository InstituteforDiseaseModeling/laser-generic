Building and Running the Documentation Docker Image
==================================================

# Build the Image

Run this command from the repository root (where
`laserdocs_sphinx/Dockerfile` is located):

``` shell
docker build -t laser-docs -f laserdocs_sphinx/Dockerfile .
```

# Run the Container to Build HTML Docs

``` shell
docker run --rm -v "$(pwd)":/docs -w /docs/sphinx laser-docs:latest html
```

# Run the Container to Build PDF (LaTeX) Docs

``` shell
docker run --rm -v "$(pwd)":/docs -w /docs/sphinx laser-docs:latest latexpdf
```

# Additional Notes

- The built documentation will appear in `sphinx/_build/html/` or
  `sphinx/_build/latex/`.
- The container will automatically run the Markdown-to-rst conversion
  step before building the docs.
