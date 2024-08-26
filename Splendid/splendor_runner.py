# INFORMATION ------------------------------------------------------------------------------------------------------- #

# Author:  Steven Spratley, extending code by Guang Ho and Michelle Blom
# Date:    09/08/2021
# Purpose: Implements "Splendor" for the COMP90054 competitive game environment

# IMPORTS ------------------------------------------------------------------------------------------------------------#

import sys
import os
import importlib
import traceback
import datetime
import time
import pickle
import random
from Splendor.splendor_model import SplendorGameRule as GameRule
from Splendor.splendor_displayer import TextDisplayer,GUIDisplayer
from template import Agent as DummyAgent
from game import Game, GameReplayer
from optparse import OptionParser

# CONSTANTS ----------------------------------------------------------------------------------------------------------#

error_index   = ["ruby_team_load","citrine_team_load","beryl_team_load","amethyst_team_load"]
DEFAULT_AGENT = "agents.generic.random"
NUM_AGENTS    = 2

# CLASS DEF ----------------------------------------------------------------------------------------------------------#

def loadAgent(file_list,name_list,superQuiet = True):
    agents = [None]*len(file_list)
    load_errs = {}
    for i,agent_file_path in enumerate(file_list):
        agent_temp = None
        try:
            mymodule = importlib.import_module(agent_file_path)
            agent_temp = mymodule.myAgent(i)
        except (NameError, ImportError, IOError):
            print('Error: Agent at "' + agent_file_path + '" could not be loaded!', file=sys.stderr)
            traceback.print_exc()
            pass
        except:
            traceback.print_exc()
            pass

        # if student's agent does not exist, use random agent.
        if agent_temp != None:
            agents[i] = agent_temp
            if not superQuiet:
                print ('Agent {} team {} agent {} loaded'.format(i,name_list[i],file_list[i]))
        else:
            agents[i] = DummyAgent(i)
            load_errs[error_index[i]] = '[Error] Agent {} team {} agent {} cannot be loaded'\
                .format(i,name_list[i],".".join((file_list[i]).split(".")[-2:]))
        
    return agents, load_errs


class HidePrint:
    # setting output stream
    def __init__(self,flag,file_path,f_name):
        self.flag = flag
        self.file_path = file_path
        self.f_name = f_name
        self._original_stdout = sys.stdout

    def __enter__(self):
        if self.flag:
            if not os.path.exists(self.file_path):
                os.makedirs(self.file_path)
            sys.stdout = open(self.file_path+"/log-"+self.f_name+".log", 'w')
            sys.stderr = sys.stdout
        else:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = sys.stdout

    # Restore
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = sys.stdout


def run(options,valid_game,msg):
    displayer = GUIDisplayer(options.half_scale, options.delay, options.no_highlighting)
    if options.textgraphics:
        displayer = TextDisplayer()
    elif options.quiet or options.superQuiet:
        displayer = None
    agents_names = [options.rubyName, options.citrineName]
    for i in range(len(agents_names)):
        agents_names[i] = agents_names[i].replace(" ","_")

    # if random seed is not provide, using timestamp
    if options.setRandomSeed == 90054:
        random_seed = int(str(time.time()).replace('.', ''))
    else:
        random_seed = options.setRandomSeed
    
    # make sure random seed is traceable
    random.seed(random_seed)
    seed_list = [random.randint(0,int(1e10)) for _ in range(1000)]
    seed_idx = 0

    num_of_warning = options.numOfWarnings
    file_path = options.output

    if options.replay != None:
        if not options.superQuiet:
            print('Replaying recorded game %s.' % options.replay)
        replay_dir = options.replay
        replay = pickle.load(open(replay_dir,'rb'),encoding="bytes")
        GameReplayer(GameRule,replay,displayer).Run()
    else: 
        games_results = [tuple([0]*NUM_AGENTS for i in range(3))]
        results = {"succ":valid_game}
        for game_num in range(options.multipleGames):
            filelist = [options.ruby, options.citrine]
            agents,load_errs = loadAgent(filelist, agents_names, superQuiet=options.superQuiet)
            is_load_err = False
            for i,err in load_errs.items():
                msg += "{} {}\n".format(i,err)
                if not options.superQuiet:
                    print(i,err)
                is_load_err = True
        
            random_seed=seed_list[seed_idx]
            seed_idx += 1

            if is_load_err:
                results["load_errors"] = load_errs
                results["succ"]=False
                valid_game = False

            f_name = agents_names[0]
            for name in agents_names[1:]:
                f_name += '-vs-'+name
            f_name += "-"+datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S-%f")
            
            gr = Game(GameRule,
                        agents,
                        num_of_agent = NUM_AGENTS,
                        seed=random_seed,
                        time_limit=options.warningTimeLimit,
                        warning_limit=num_of_warning,
                        displayer=displayer,
                        agents_namelist=agents_names,
                        interactive=options.interactive)
            if not options.print:
                with HidePrint(options.saveLog,file_path,f_name):
                    print("Following are the print info for loading:\n{}\n".format(msg))
                    print("\n-------------------------------------\n")
                    print("Following are the print info from the game:\n")
                    if valid_game:          
                        replay = gr.Run()
                    else:
                        print("Invalid game. No game played.\n")
            else:
                print("Following are the print info for loading:\n{}\n".format(msg))
                print("\n-------------------------------------\n")
                print("Following are the print info from the game:\n")
                if valid_game:      
                    replay = gr.Run()
                else:
                    print("Invalid game. No game played.\n")
                    
            if valid_game:
                scores,totals,wins = games_results[len(games_results)-1]
                new_scores = []
                new_totals = []
                new_wins = []
                
                #Record scores.
                for i in range(NUM_AGENTS):
                    new_scores.append(replay["scores"][i])
                max_score = max(new_scores)
                
                #Order agent IDs and scores by their ranks this game. Ranks is a list of ranks (int) in player order.
                #Ranks record ties, so if 2 or more agents achieve the same score, they also achieve the same rank.
                ids,scores = list(zip(*sorted(replay["scores"].items(), key=lambda x : x[1], reverse=True)))
                ranks = []
                for agent_id,score in zip(ids,scores):
                    ranks.append((agent_id, scores.index(score) + 1))
                ranks = [i[1] for i in sorted(ranks, key=lambda x : x[0])]

                #Update totals and wins (cumulative).
                num_first = 0
                for i in range(NUM_AGENTS):
                    new_totals.append(totals[i]+new_scores[i])
                    if new_scores[i]==max_score:
                        num_first += 1
                        new_wins.append(wins[i]+1)
                    else:
                        new_wins.append(wins[i])

                if not options.superQuiet:
                    print("Result of game ({}/{}):".format(game_num+1, options.multipleGames))
                    for i in range(NUM_AGENTS):
                        print("    {} earned {} points.".format(agents_names[i],new_scores[i]))
    
                games_results.append((new_scores,new_totals,new_wins))
                if options.saveGameRecord:
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    if not options.superQuiet:
                        print("Game ({}/{}) has been recorded!".format(game_num+1,options.multipleGames))
                    record = pickle.dumps(replay)
                    with open(file_path+"/replay-"+f_name+".replay",'wb') as f:
                        f.write(record)
        
        if valid_game:
            scores,totals,wins = games_results[len(games_results)-1]
            avgs = []
            win_rates = []
            for i in range(NUM_AGENTS):
                avgs.append(totals[i]/options.multipleGames)
                win_rates.append(wins[i]/options.multipleGames*100)

            if not options.superQuiet:
                print("Over {} games:".format(options.multipleGames))
                for i in range(NUM_AGENTS):
                    print("    {} earned {:.2f} on average and won {} games ({:.2f})%."\
                          .format(agents_names[i],avgs[i],wins[i],win_rates[i]))

            # return results as statistics
            results["totals"] = totals
            results["wins"] = wins
            results["win_rates"] = win_rates
            results["names"] = agents_names
            results["fileName"] = f_name
            results["load_errs"] = load_errs
            results["succ"] = True

        return results


def loadParameter():

    """
    Processes the command used to run Splendor from the command line.
    """
    usageStr = """
    USAGE:      python runner.py <options>
    EXAMPLES:   (1) python runner.py
                    - starts a game with four random agents.
                (2) python runner.py -c MyAgent
                    - starts a fully automated game where Citrine team is a custom agent and the rest are random.
    """
    parser = OptionParser(usageStr)
    parser.add_option('-r', '--ruby', help='Ruby team agent file', default=DEFAULT_AGENT)
    parser.add_option('-c', '--citrine', help='Citrine team agent file', default=DEFAULT_AGENT) 
    parser.add_option('--rubyName', help='Ruby agent name', default='Ruby')
    parser.add_option('--citrineName', help='Citrine agent name', default='Citrine') 
    parser.add_option('-t','--textgraphics', action='store_true', help='Display output as text only (default: False)', default=False)
    parser.add_option('-q','--quiet', action='store_true', help='No text nor graphics output, only show game info', default=False)
    parser.add_option('-Q', '--superQuiet', action='store_true', help='No output at all', default=False)
    parser.add_option('-w', '--warningTimeLimit', type='float',help='Time limit for a warning of one move in seconds (default: 1)', default=1.0)
    parser.add_option('--startRoundWarningTimeLimit', type='float',help='Time limit for a warning of initialization for each round in seconds (default: 5)', default=5.0)
    parser.add_option('-n', '--numOfWarnings', type='int',help='Num of warnings a team can get before fail (default: 3)', default=3)
    parser.add_option('-m', '--multipleGames', type='int',help='Run multiple games in a roll', default=1)
    parser.add_option('--setRandomSeed', type='int',help='Set the random seed, otherwise it will be completely random (default: 90054)', default=90054)
    parser.add_option('-s','--saveGameRecord', action='store_true', help='Writes game histories to a file (named by teams\' names and the time they were played) (default: False)', default=False)
    parser.add_option('-o','--output', help='output directory for replay and log (default: output)',default='output')
    parser.add_option('-l','--saveLog', action='store_true',help='Writes agent printed information into a log file(named by the time they were played)', default=False)
    parser.add_option('--replay', default=None, help='Replays a recorded game file by a relative path')
    parser.add_option('--delay', type='float', help='Delay action in a play or replay by input (float) seconds (default 0.1)', default=0.1)
    parser.add_option('-p','--print', action='store_true', help='Print all the output in terminal when playing games, will diable \'-l\' automatically. (default: False)', default=False)
    parser.add_option('--half-scale', action='store_true', help='Display game at half-scale (default is 1920x1080)', default=False)
    parser.add_option('--interactive', action='store_true', help="Gives the user control over the Citrine agent's actions", default=False)   
    parser.add_option('--no-highlighting', action='store_true', help="Disables highlighting of potential card purchases", default=False)

    options, otherjunk = parser.parse_args(sys.argv[1:] )
    assert len(otherjunk) == 0, "Unrecognized options: " + str(otherjunk)
    if options.interactive:
        options.citrineName = 'Human'
    return options

# MAIN ---------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':

    """
    The main function called when advance_model.py is run
    from the command line:

    > python runner.py

    See the usage string for more details.

    > python runner.py --help
    """
    msg = ""
    options = loadParameter()
    run(options,True,msg)

# END FILE -----------------------------------------------------------------------------------------------------------#
