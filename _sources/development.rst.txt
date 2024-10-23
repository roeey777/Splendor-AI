Development
-----------

This repository uses multiple automatic tools in order to reach a high standard.
The tools used are:

#. ``ruff`` as a linter & formatter for the code.
#. ``mypy`` is used for type validation.

There is also pre-commit hook which uses executes ``ruff`` automatically before every commit.
In order to activate this hook one should execute the following command:

.. code-block:: bash

   pre-commit install

This tool installs into ``.git/hooks/`` the hook which activates ``ruff``.

There is an issue when it comes to using ``mypy`` as a pre-commit hook as well, for more details you can look at `this issue <https://github.com/python/mypy/issues/13916>`_.

In earlier stages of this project, we've used ``black``, ``isort`` & ``pylint``, however we've seen that ``ruff`` offers the same functionality (and more) but takes only a fraction of the execution time (real world, wall clock time).
Even though we've ditched ``isort`` & ``black`` in favor of ``ruff`` you can still use them since ``ruff`` maintains compatibility with those tools.
You could also still use ``pylint`` as it isn't checks the same rules as ``ruff`` and the project still maintains a high score of over 9.9 with ``pylint``.
