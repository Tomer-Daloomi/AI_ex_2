import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        # "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        score = successor_game_state.score

        # potentially useful characteristics for determining the overall score by which we would
        # like to advance :

        max_tile = successor_game_state.max_tile
        min_tile = np.min(board[np.nonzero(board)])
        num_of_empty_tiles = len(successor_game_state.get_empty_tiles())

        return score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        # we'll use this list to collect the minmax returned value for the best score
        best_moves = []
        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()
        # Choose one of the best actions
        successor_states = [game_state.generate_successor(action=action) for action in legal_moves]

        for successor in successor_states:
            depth_counter = 1
            best_moves.append(self.minimax_val(successor, depth_counter))

        best_choice = max(best_moves)
        best_indices = [index for index in range(len(best_moves)) if best_moves[index] == best_choice]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]

    def minimax_val(self, game_state, depth_counter):
        """
        executes the minimax recursive algorithm over a given game state, and returns the value
        of the highest leaf of the tree out of the given depth
        mashehu
        :param game_state:
        :param depth_counter:
        :return:
        """

        if depth_counter == self.depth * 2 - 1 or game_state.get_legal_actions(0) == [] or \
                        game_state.get_legal_actions(1) == []:
            return score_evaluation_function(game_state)

        agent = depth_counter % 2

        legal_moves = game_state.get_agent_legal_actions()

        successor_states = [game_state.generate_successor(action=action) for action in
                            legal_moves]

        if agent == 0:
            return max([self.minimax_val(game_state, depth_counter + 1) for game_state in
                              successor_states])

        if agent == 1:
            return min([self.minimax_val(game_state, depth_counter + 1) for game_state in
                               successor_states])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = better_evaluation_function
