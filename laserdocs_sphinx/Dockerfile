# Build docs for LASER-GENERIC
FROM sphinxdoc/sphinx-latexpdf:latest

# Install system pandoc and jupyter
RUN apt-get update && apt-get install -y pandoc && rm -rf /var/lib/apt/lists/*

# Install extra Sphinx extensions you use
RUN pip install --no-cache-dir \
      myst-parser[linkify] \
      sphinx-autodoc-typehints \
      sphinx-rtd-theme \
      sphinx-autoapi \
      jupyter \
      nbconvert

#ENTRYPOINT ["make"]
#ENTRYPOINT ["sh", "-c", "python3 scripts/convert_docs.py && make $@"]

ENTRYPOINT sh -c "python3 scripts/convert_docs.py && python3 scripts/convert_notebooks.py && make \"$@\""

# Default workdir inside the container
WORKDIR /docs/sphinx

# Default command (may be overridden)
#CMD ["make", "html"]
CMD ["html"]
