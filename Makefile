SOURCE_DIR := src/splendor
DOCS_SOURCE_DIR := docs/source
DOCS_BUILD_DIR := docs/_build/html
COMMIT_ID := $(shell git rev-parse HEAD)
COMMIT_MSG := $(addprefix Documentation for commit ,$(COMMIT_ID))

.PHONY: format
format:
	ruff format .

.PHONY: lint
lint:
	ruff check --preview .
	mypy .
	pylint src/

.PHONY: pre-commit
pre-commit:
	pre-commit install
	pre-commit run --all-files

.PHONY: docs
docs:
	splendor --version  # ensure splendor is installed.
	sphinx-apidoc --output-dir $(DOCS_SOURCE_DIR) $(SOURCE_DIR) --force
	sphinx-build $(DOCS_SOURCE_DIR) $(DOCS_BUILD_DIR)

.PHONY: publish-docs
publish-docs: docs
	pre-commit uninstall
	cd $(DOCS_BUILD_DIR) && git add -A . && git commit -sm "$(COMMIT_MSG)." && git push origin gh-pages
	pre-commit install

.PHONY: clean
clean:
	make -C docs/ clean
