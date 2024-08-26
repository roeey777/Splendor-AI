# UoM COMP90054 Contest Project - Team greedisgood

This Wiki can be used as external documentation to the project.

1. [Home and Introduction]()
2. [Design Choices (Offense/Defense)](Design-Choices)

   2.1 [Approach One](AI-Method-1)

   2.2 [Approach Two](AI-Method-2)
3. [Evolution and Experiments](Evolution)
4. [Conclusions and Reflections](Conclusions-and-Reflections)

# Youtube presentation
https://www.youtube.com/watch?v=ndJXylCSGCE


## Team Members

List here the full name, email, and student number for each member of the team:

* Rebonto Zaman Joarder - zamanr@student.unimelb.edu.au - 911679
* Lee Guo Yi - glee3@student.unimelb.edu.au - 893164
* Jiacheng Zhang - jiachzhang2@student.unimelb.edu.au - 1218000

## Introduction

For this contest we aimed to start simple by using a search algorithm. Firstly, we began by getting acquainted with the game of Splendor and exploring some of the popular strategies out there. We concluded that we could find ways to calculate optimal strategies by just observing some of future states we could be in. However, it was hard for us to take note of all the possible moves that we could make, and which move would indeed be the correct one to make to maximize our chances of winning.

We realised that we were asking one main question, "what is the optimal to move to make given any state?". This served as the inspiration for our agents. We used a computer's calculation and memory prowess to essentially improve on an already existing human process. 

Using search algorithms, we were able to pre-determine the sort of path (of moves) we wanted our algorithm to make by hand coding a heuristic function with experimented weights. And to extend upon that we wanted to know the true weights as played out a few times by our AI to figure out what those optimal weights should be.

Thus, this wiki documents our process, challenges, and design processes to be able to obtain our agent for this test.
