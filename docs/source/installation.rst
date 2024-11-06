Installation of Splendor
------------------------

There are 2 possible ways to install the requirements of splendor.

#. using ``conda``.
#. using ``pip``.


Install Splendor using ``conda``:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute the following (in the repo’s top directory):

::

   conda env create -f environment.yaml
   conda activate splendor
   pip install .

Install Splendor using ``pip``:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute the following (in the repo’s top directory):

::

   pip install -r requirements.txt
   pip install .

Verify The Installation
~~~~~~~~~~~~~~~~~~~~~~~

::

   splendor --help
