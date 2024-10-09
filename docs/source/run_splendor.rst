Run the game
------------

Execute the following command for help message (location is no longer
relevant):

::

   splendor --help

Interactive mode
~~~~~~~~~~~~~~~~

::

   splendor --interactive

Specify Opponents
~~~~~~~~~~~~~~~~~

There are a few option for specifying agents: 1. The specified agent is
part of ``splendor``. 2. The specified agent is **not** part of
``splendor``, however he is installed as a part of a different package.
3. The specified agent is **not** part of ``splendor`` and he is **not
installed** as a part of a different package.

We’ll now address each case.

Case #1 - Specifying Opponents from ``splendor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Whenever you wish to invoke/use a specific agent (from ``splendor``) you
need to specify the **absolute import path**. The absolute import path
must be specified **regardless of the working directory**.

::

   splendor -a splendor.agents.generic.random,splendor.agents.generic.first_move --agent_names=random,first_move

Case #2 - Specifying Opponents not from ``splendor`` (installed via other package)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let’s assume we’ve installed a package called ``external`` and there is
an agent called ``best`` whithin ``external.agents`` and we want to
flesh out this agent againt the random agent we would execute the
following command:

::

   splendor -a splendor.agents.generic.random,external.agents.best --agent_names=random,external

Case #3 - Specifying Opponents not from ``splendor`` (not installed via other package)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let’s assume we want to use an agent called agent_in_my_cwd which isn’t
part of ``splendor`` nor installed via another package. We would utilize
the fact that the game adds the current working directory to the module
search path when loading agents. So we would act as follows:

::

   cd
   splendor -a agent_in_my_cwd,another_agent_in_my_cwd --agent_names=external_1,external_2

Note - use with caution:
''''''''''''''''''''''''

By default the game adds the current working directory to the module
search path when loading agents. This can be disabled by providing the
flag ``--absolute-imports`` however this would deny the usage of agents
which aren’t part of ``splendor`` without installing them as part of
other package.

Explanation
^^^^^^^^^^^

1. the ``-a`` flag is used to specify which agents to load, this must be
   comma seperated values, where each value must be an import path of
   the agent to be loaded. Moreover each of those agent must inherit
   from ``splendor.template.Agent`` and must call their agent (or a
   factory) by the following name - ``myAgent``.
2. the ``--agent_names=`` is another comma seperated argument which
   specifies the names given to each agent. The number of agents to be
   loaded is determined by the amount of names given, when there are
   more names listed than agents listed the game will automatically load
   random agents to fill the void.

Without GUI (Textual Mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~

just add the ``-t`` option, for example:

::

   splendor -a splendor.agents.generic.random,splendor.agents.generic.first_move --agent_names=random,first_move -t

Using Our Agents
~~~~~~~~~~~~~~~~

Interactively play against our trained agents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interactively play against the trained genetic algorithm agent:

::

   splendor -a splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent --agent_names=genetic,human --interactive

Interactively play against the trained PPO agent:

::

   splendor -a splendor.agents.our_agents.ppo.ppo_agent --agent_names=ppo,human --interactive

Let them play by them selves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let the genetic algorithm agent play against minimax (with alpha-beta
pruning) agent:

::

   splendor -a splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent,splendor.agents.our_agents.minmax --agent_names=genetic,minimax

Let the genetic algorithm agent play against minimax (with alpha-beta
pruning) agent for 10 consecutive games (only text display):

::

   splendor -a splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent,splendor.agents.our_agents.minmax --agent_names=genetic,minimax -t -m 10

Let the PPO agent play against minimax (with alpha-beta pruning) agent for 10 consecutive games (only text display):

::

        splendor -a splendor.agents.our_agents.ppo.ppo_agent,splendor.agents.our_agents.minmax –agent_names=ppo,minimax -t -m 10

