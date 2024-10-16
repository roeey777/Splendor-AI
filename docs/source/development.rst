Development
-----------

This repository uses multiple automatic tools in order to reach a high standard.
The tools used are:

#. ``black`` as a formatter for the code.
#. ``isort`` is used for organizing the imports.
#. ``mypy`` is used for type validation.
#. ``pylint`` as a linter.

There is also pre-commit hook which uses executes ``black``, ``isort`` & ``pylint`` automatically before every commit.
In order to activate this hook one should execute the following command:

.. code-block:: bash

   pre-commit install

This tool installs into ``.git/hooks/`` the hook which activates ``black``, ``isort`` & ``pylint``.

There is an issue when it comes to using ``mypy`` as a pre-commit hook as well, for more details you can look at `this issue <https://github.com/python/mypy/issues/13916>`_.
