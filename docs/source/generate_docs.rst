Generate The Documentation
--------------------------

This project uses ``sphinx`` in order to generate it's documentation & hosts them on `Github Pages <https://roeey777.github.io/Splendor-AI/>`_.
There are a few steps for generating & hosting documentation.

Automatic generation of documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At first you should verify that you are using the ``splendor`` environment & the package is installed.

.. code-block:: bash

   conda activate splendor
   splendor --version

If not installed please refer to the installation page.

Afterwards you should execute the following commands:

.. code-block:: bash

   # from docs/
   make clean
   
   # move to the repo top directory
   cd ..
   sphinx-apidoc --output-dir docs/source/ src/splendor --force
   sphinx-build docs/source/ docs/_build/html

And now your documentation is built!
You can inspect it as follows:

.. code-block:: bash

   firefox docs/_build/html/index.html

Publishing the Documentation to Github Pages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now since you've used the ``make clean`` command, the directory ``docs/_build/html`` was automatically added
to git worktree for the branch ``gh-pages`` which is the default branch `Github <github.com>` uses for the pages feature.
All that is left to do is as follows:

.. code-block:: bash

   cd docs/_build/html

Now you need to verify that your working on ``gh-pages`` branch, this can be validated as follows:

.. code-block:: bash

   # from docs/_build/html
   git branch

After this verification we can add all the new documentation.

.. code-block:: bash

   # from docs/_build/html
   git add -A .
   git commit -sm "update documentation"
   git push origin gh-pages

And Your'e Done!

