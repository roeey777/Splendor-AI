Developing an Agent
-------------------

In order for the game to properly load your agent one must install the
agent, there are several ways to do so: 1. create a new agent within
``src/splendor/agents`` and when installing splendor your agent will be
installed as well. (i.e. when invoking ``pip install .``) 2. create a
new package and develop your agent there and then install it. 3. create
a new agent within ``src/splendor/agents`` and **ONLY DURING
DEVELOPMENT** install splendor by using ``pip install -e .`` (instead of
the ``pip install .``) which allows you to edit and adjust your agent
as you please without the necessity to re-install the package.

Training Our Agents:
~~~~~~~~~~~~~~~~~~~~

Training The Genetic Algorithm Agent:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to train the genetic algorithm agent with the following
hyper-parameters: 1. Specify the population size in each generation to
be 24 (should be a multiple of 12). 2. Train for 20 generations. 3. Fix
the mutation rate chance to be 0.1(%). 4. Use a fixed random seed. Use
the following command:

.. code-block:: bash

   evolve --population-size 24 --generations 20 --mutation-rate 0.1 --seed 1234

Training The PPO Agent:
~~~~~~~~~~~~~~~~~~~~~~~

In order to train the PPO agent you should run the following command:

.. code-block:: bash

   ppo

This command will train the PPO agent with the default training
hyper-parameters.

.. code-block:: bash

   ppo --device cuda --working-dir runs --transfer-learning --opponent minimax

This command will use GPU during it's training, it will use the installed weights as initialization of the network
and the PPO will be trained against MiniMax. Furthermore all the generated files (weights stored in ``.pth`` files and ``stats.csv``) will be generated within the directory ``runs/``.

There are multiple available architectures for the neural network to be used by the PPO agent, for instance:

#. MLP - Multi-Layered Perceptron, also known as Fully-Connected Feed-Forward network.
#. Self-Attention and then MLP.
#. GRU (Gated Recurrent Unit) and then MLP.
#. LSTM (Long Short-Term Memory) and then MLP.

By default the MLP architecture will be used, however you can decide to train the PPO with a different architecture via the ``--architecture`` or via it's shortcuts ``-a`` and ``--arch``, here is an example:

.. code-block:: bash

   ppo --device cpu --opponent random --architecture gru

There are also multiple opponents available to be trained or evaluated against, such as:

#. random
#. minimax
#. itself
#. PPO (with MLP)
#. PPO (with Self-Attention)
#. PPO (with GRU)
#. PPO (with LSTM)

``SplendorEnv`` - an OpenAI ``gym`` compatible simulator for the game Splendor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`gym`_ or currently `gymnasium`_ (Previously maintained by `OpenAI`_ and now by the `Farama Foundation <https://farama.org/>`_) is a framework providing an API standard for reinforcement learning with a diverse collection of reference environments.

.. _gym: https://www.gymlibrary.dev/

.. _gymnasium: https://gymnasium.farama.org/index.html

.. _OpenAI: https://openai.com/

We’ve made a custom ``gym.Env`` and registered it as one of ``gym``
environments. This would come in handy when training agent such as DQN
or PPO.

How to create an instance of ``SplendorEnv``:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. import ``gymnasium`` - ``import gymnasium as gym``.
2. registering ``SplendorEnv`` to ``gym`` -
   ``import splendor.Splendor.gym``
3. define the opponents:

When creating an instance of ``SplendorEnv`` you should tell it which
agents will be used as opponents to you (the one who uses the env.). For
the following example we’ll use a single random agent as an opponent.

.. code-block:: python

   from splendor.agents.generic.random import myAgent

   opponents = [myAgent(0)]

4. creating the environment:

.. code-block:: python

   env = gym.make("splendor-v1", agents=opponents)

Custom features of ``SplendorEnv``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. every call to ``env.step(action)`` simulate (by using
   ``SplendorGameRule``) the turns of all the opponents.
2. when calling ``env.reset()`` ``SplendorEnv`` will return the feature
   vector of the initial state AND the turn of our agent via the second
   variable (the ``dict``) which will have a key called ``my_turn``.
3. ``SplendorEnv`` have several custom properties:

   1. ``state`` - the actual ``SplendorState`` - not the feature vector.
   2. ``my_turn`` - the turn of the agent, same as the value returned by
      ``env.reset()``.

4. ``SplendorEnv`` have several custom methods:

   1. ``get_legal_actions_mask`` - a method for getting a mask vector
      which masks all the illegal action of
      ``splendor.Splendor.gym.envs.actions.ALL_ACTIONS``.

You can access those like this:

.. code-block:: python

   env.unwrapped.my_turn

