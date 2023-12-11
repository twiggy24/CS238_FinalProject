import random
import math
from copy import deepcopy
import time
import matplotlib.pyplot as plt

validColumns = set((0, 1, 2, 3, 4, 5, 6))
EMPTY_GRID = [[0] * 7 for i in range(6)]
EMPTY_COUNTS = [0] * 7
VALID_PLAYS_IN_BEGINNING = set((0, 1, 2, 3, 4, 5, 6))
# 1 is you, 2 is the AI
PLAYERS = [1, 2]
# 0 to signify nothing significant has occurred, 1 to signify you won, 2 to signify AI won, and 3 to signify a draw.
GAME_OUTCOMES = [0, 1, 2, 3]
ROWS = 6
COLS = 7
EXPLORATION_CONSTANT = math.sqrt(2)
INFINITY = float('inf')


# Class to represent Connect 4 grid as the state.
class Connect4:
    def __init__(self):
        self.grid = deepcopy(EMPTY_GRID)
        self.currCounts = deepcopy(EMPTY_COUNTS)
        self.validPlays = deepcopy(VALID_PLAYS_IN_BEGINNING)
        self.firstTurnPlayer = random.choice(PLAYERS)
        self.currPlayer = self.firstTurnPlayer
        self.prevPlay = ()

    # Function to make play and update grid state.
    def makePlay(self, col):
        row = self.currCounts[col]
        self.grid[row][col] = self.currPlayer
        self.prevPlay = row, col
        self.currCounts[col] += 1
        if self.currCounts[col] == ROWS:
            self.validPlays.remove(col)
        self.swapPlayers()

    # You make a play.
    def makeHumanPlay(self):
        while True:
            coll = input("Make a play: ")
            if not coll.isdigit():
                print("Invalid play! Try again")
            else:
                coll = int(coll) - 1
                if coll not in self.validPlays:
                    print("Invalid play! Try again")
                else:
                    self.makePlay(coll)
                    return coll

    # Robot makes a random play.
    def makeRandomPlay(self):
        # Chooses random available column
        coll = random.choice(self.getValidPlays())
        self.makePlay(coll)
        return coll

    # Player swaps after a turn was taken.
    def swapPlayers(self):
        if self.currPlayer == 1:
            self.currPlayer = 2
        else:
            self.currPlayer = 1

    # Return a list of columns that are not full.
    def getValidPlays(self):
        return list(self.validPlays)

    # Checks if a player has won.
    def fourInARow(self):
        for row in range(ROWS):
            for coll in range(COLS - 3):
                if all(self.grid[row][coll + i] == self.grid[row][coll] and (self.grid[row][coll] == 1 or self.grid[row][coll] == 2) for i
                       in range(4)):
                    return True
        for coll in range(COLS):
            for row in range(ROWS - 3):
                if all(self.grid[row + i][coll] == self.grid[row][coll] and (self.grid[row][coll] == 1 or self.grid[row][coll] == 2) for i
                       in range(4)):
                    return True
        for row in range(ROWS):
            for coll in range(COLS):
                if row <= ROWS - 4 and coll <= COLS - 4:
                    if all(self.grid[row + i][coll + i] == self.grid[row][coll] and (self.grid[row][coll] == 1 or self.grid[row][coll] == 2)
                           for i in range(4)):
                        return True
                if row <= ROWS - 4 and coll >= 3:
                    if all(self.grid[row + i][coll - i] == self.grid[row][coll] and (self.grid[row][coll] == 1 or self.grid[row][coll] == 2)
                           for i in range(4)):
                        return True
        return False

    # Checks if every column is filled.
    def gridFilled(self):
        if len(self.validPlays) == 0:
            return True
        return False

    # Checks if a player has won, and if so, returns the player that won.
    def hasPlayerWon(self):
        if self.fourInARow():
            return self.grid[self.prevPlay[0]][self.prevPlay[1]]
        return 0

    # Checks if the game has finished.
    def gameDone(self):
        return self.hasPlayerWon() or self.gridFilled()

    # If the game is ended, find out if it is a tie, and if not, return the winner.
    def getGameResults(self):
        if self.gridFilled() and (not self.hasPlayerWon()):
            return 3
        if self.hasPlayerWon() == 1:
            return 1
        else:
            return 2

    # Draws the current Connect 4 grid.
    def drawBoard(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')
        # Draw game board
        for x in range(COLS):
            for y in range(ROWS):
                if self.grid[y][x] == 1:  # Blue for player 1
                    color = 'blue'
                elif self.grid[y][x] == 2:  # Red for player 2
                    color = 'red'
                else:  # Empty
                    color = 'white'
                # Draw cell and chip (or empty slot)
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='yellow', edgecolor='black'))
                circle = plt.Circle((x, y), 0.45, color=color)
                ax.add_patch(circle)
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, 5.5)
        ax.axis('off')
        plt.draw()
        plt.pause(10)  # Pause to update board
        plt.close()


# Class to represent a Node for state, action pair in the Monte Carlo Tree.
class ActionNode:
    def __init__(self, action, prevStateNode):
        self.action = action  # The action taken in this node.
        self.prevStateNode = prevStateNode  # Parent node in tree
        self.nextActionNodes = {}  # Holds child nodes for subsequent actions
        self.Nsa = 0  # Number of times this action has been visited in simulations
        self.Q = 0  # Total value (reward) accumulated from simulations through this action.

    # Gets value of action node
    def getNodeValue(self):
        if self.Nsa == 0:
            # Prioritize unvisited nodes
            return INFINITY
        else:
            # Use UCB1 exploration heuristic for nodes that have been visited.
            return self.ucb1()

    # Calculates the UCB1 exploration heuristic for the action node
    def ucb1(self):
        Qsa = self.Q / self.Nsa  # Average reward per visit of this action
        Ns = math.log(self.prevStateNode.Nsa)  # Log of the total visits to parent node
        explorationBonus = math.sqrt(Ns / self.Nsa)   # Exploration bonus to encourage trying less visited nodes
        return Qsa + (EXPLORATION_CONSTANT * explorationBonus)

    # Add child nodes to this node for each possible next action
    def addChildNodes(self, nextActions):
        for node in nextActions:
            # Map each child node's action to the node object itself
            self.nextActionNodes[node.action] = node


# Class to represent Monte Carlo tree.
class MonteCarlo:
    def __init__(self, state):
        # state, action pair
        self.state = deepcopy(state)
        self.root = ActionNode(None, None)

    # This function selects the next action node to explore in the Monte Carlo tree.
    def selectActionNode(self):
        node = self.root
        # Work with a copy of the current state to avoid modifying the actual game state.
        state = deepcopy(self.state)
        # Loop until an unexplored node is found or a new node is added.
        while len(node.nextActionNodes):
            nextActions = node.nextActionNodes.values()
            # Start with the lowest possible value
            maxValue = -INFINITY
            # Iterate through each node to find the one with the highest value
            for node in nextActions:
                # Get the value of the current node
                nodeValue = node.getNodeValue()
                if nodeValue > maxValue:
                    # Update maxValue if the current node's value is higher
                    maxValue = nodeValue
            # Collect nodes with the highest value.
            topNodes = []
            for node in nextActions:
                if node.getNodeValue() == maxValue:
                    topNodes.append(node)
            # Randomly choose among the top nodes for exploration.
            node = random.choice(topNodes)
            state.makePlay(node.action)
            # If node has not been explored, it's selected for expansion
            if node.Nsa == 0:
                return node, state
        # assume game is still going
        play = 1
        # check if game is over
        if state.gameDone():
            play = 0
        else:
            # Add new child nodes for each valid action if the game is not over
            childActions = []
            for nextPlay in state.getValidPlays():
                childActions.append(ActionNode(nextPlay, node))
            node.addChildNodes(childActions)
        # Choose a new action to explore further.
        if play == 1:
            node = random.choice(list(node.nextActionNodes.values()))
            state.makePlay(node.action)
        return node, state

    # This function conducts the Monte Carlo tree within the given time limit by
    # repeatedly selecting nodes to explore, simulating games from those nodes, rollouts,
    # and back propagating the results.
    def performMonteCarloSearch(self, timeLimit):
        startTime = time.time()
        # Perform search under time constraint
        while time.time() - startTime < timeLimit:
            node, state = self.selectActionNode()
            while not state.gameDone():
                state.makePlay(random.choice(state.getValidPlays()))
            result = state.getGameResults()
            # Determine reward based on game results.
            if state.currPlayer == result: 
                reward = 0
            else: 
                reward = 1
            # Back propagate the results through the path in the tree,
            # updating nodes with simulation results from the current node to the root.
            while node is not None:
                # Increment the visit count
                node.Nsa += 1
                # Update total reward
                node.Q += reward
                #  Move up the tree
                node = node.prevStateNode
                # Adjust the reward for the parent node's perspective
                if result == 3:
                    reward = 0
                else:
                    reward = 1 - reward

    # This function determines the best play to make from the current state.
    def killerMove(self):
        nextActions = self.root.nextActionNodes.values()
        max_visits = 0
        # Looks at all next possible actions from the root and select the one with the most visits
        for node in nextActions:
            if node.Nsa >= max_visits:
                max_visits = node.Nsa
        maxNodes = []
        for node in nextActions:
            if node.Nsa == max_visits:
                maxNodes.append(node)
        # If multiple nodes have the same maximum number of visits, it randomly selects one of them.
        return random.choice(maxNodes).action

    # This function updates the Monte Carlo tree and game state after a move is made.
    def play(self, action):
        # If the action is already in the tree,
        # it updates the tree to reflect this move by moving to the corresponding child node.
        if action in self.root.nextActionNodes:
            self.state.makePlay(action)
            self.root = self.root.nextActionNodes[action]
        else:
            # If the action is new, it resets the tree with a new root node
            self.state.makePlay(action)
            self.root = ActionNode(None, None)


# Play game of Connect4 against AI.
def playConnect4(AISearchTime):
    currGame = Connect4()
    monteCarlo = MonteCarlo(currGame)
    while not currGame.gameDone():
        if currGame.firstTurnPlayer == 1:
            monteCarlo.play(currGame.makeHumanPlay())
            if currGame.gameDone():
                print("You won!")
                currGame.drawBoard()
                break
            print("Wait. AI is deciding.")
            monteCarlo.performMonteCarloSearch(AISearchTime)
            col = monteCarlo.killerMove()
            print("AI chose column: ", col)
            currGame.makePlay(col)
            monteCarlo.play(col)
            if currGame.gameDone():
                print("AI won!")
                currGame.drawBoard()
                break
        else:
            print("Wait. AI is deciding.")
            monteCarlo.performMonteCarloSearch(AISearchTime)
            col = monteCarlo.killerMove()
            print("AI chose column: ", col + 1)
            currGame.makePlay(col)
            monteCarlo.play(col)
            if currGame.gameDone():
                print("AI won!")
                currGame.drawBoard()
                break
            monteCarlo.play(currGame.makeHumanPlay())
            if currGame.gameDone():
                print("You won!")
                currGame.drawBoard()
                break


# Random Robot plays Connect4 against AI.
def RandomVSAIPlayConnect4(AISearchTime):
    currGame = Connect4()
    monteCarlo = MonteCarlo(currGame)
    while not currGame.gameDone():
        if currGame.firstTurnPlayer == 1:
            coll = currGame.makeRandomPlay()
            print("Random robot chose column: ", coll + 1)
            monteCarlo.play(coll)
            if currGame.gameDone():
                print("Random robot won!")
                currGame.drawBoard()
                break
            print("Wait. AI is deciding.")
            monteCarlo.performMonteCarloSearch(AISearchTime)
            col = monteCarlo.killerMove()
            print("AI chose column: ", col)
            currGame.makePlay(col)
            monteCarlo.play(col)
            if currGame.gameDone():
                print("AI won!")
                currGame.drawBoard()
                break
        else:
            print("Wait. AI is deciding.")
            monteCarlo.performMonteCarloSearch(AISearchTime)
            col = monteCarlo.killerMove()
            print("AI chose column: ", col + 1)
            currGame.makePlay(col)
            monteCarlo.play(col)
            if currGame.gameDone():
                print("AI won!")
                currGame.drawBoard()
                break
            coll = currGame.makeRandomPlay()
            print("Random robot chose column: ", col + 1)
            monteCarlo.play(coll)
            if currGame.gameDone():
                print("Random Robot won!")
                currGame.drawBoard()
                break


def AIPlayConnect4(AISearchTime1, AISearchTime2):
    currGame = Connect4()
    monteCarlo = MonteCarlo(currGame)
    while not currGame.gameDone():
        print("Wait. AI one is deciding.")
        monteCarlo.performMonteCarloSearch(AISearchTime1)
        col1 = monteCarlo.killerMove()
        print("AI one chose column: ", col1 + 1)
        currGame.makePlay(col1)
        monteCarlo.play(col1)
        if currGame.gameDone():
            print("AI one won!")
            currGame.drawBoard()
            break
        print("Wait. AI two is deciding.")
        monteCarlo.performMonteCarloSearch(AISearchTime2)
        col2 = monteCarlo.killerMove()
        currGame.makePlay(col2)
        monteCarlo.play(col2)
        print("AI two chose column: ", col2 + 1)
        if currGame.gameDone():
            print("AI two won!")
            currGame.drawBoard()
            break


def main():
    # playConnect4(.1)
    # RandomVSAIPlayConnect4(.05)
    AIPlayConnect4(15, 16)



if __name__ == "__main__":
    main()
