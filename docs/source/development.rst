Development
-----------

This repository uses ``black`` as a formatter for the code.
There is also pre-commit hook which uses ``black`` for automation.
In order to activate this hook one should execute the following command:

.. code-black:: bash

   pre-commit install

This tool installes into ``.git/hooks/`` the hook which activates ``black``.
