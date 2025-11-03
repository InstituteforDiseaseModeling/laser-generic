# Build docs for LASER-GENERIC
FROM sphinxdoc/sphinx-latexpdf:latest

# Install extra Sphinx extensions you use
RUN pip install --no-cache-dir \
      myst-parser \
      sphinx-autodoc-typehints \
      sphinx-rtd-theme \
      sphinx-autoapi

ENTRYPOINT ["make"]

# Default workdir inside the container
WORKDIR /docs/sphinx

# Default command (may be overridden)
#CMD ["make", "html"]
CMD ["html"]
