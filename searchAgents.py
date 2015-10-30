from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import searchAgents

class GoWestAgent(Agent):
  "An agent that goes West until it can't."

  def getAction(self, state):
    "The agent receives a GameState (defined in pacman.py)."
    if Directions.WEST in state.getLegalPacmanActions():
      return Directions.WEST
    else:
      return Directions.STOP

class GoWestAgent(Agent):
  "An agent that goes West until it can't."

  def getAction(self, state):
    "The agent receives a GameState (defined in pacman.py)."
    if Directions.WEST in state.getLegalPacmanActions():
      return Directions.WEST
    else:
      return Directions.STOP

class SearchAgent(Agent):
  def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
    if fn not in dir(search):
      raise AttributeError, fn + ' is not a search function in search.py.'
    func = getattr(search, fn)
    if 'heuristic' not in func.func_code.co_varnames:
      print('[SearchAgent Class] Using Function ' + fn)
      self.searchFunction = func
    else:
      if heuristic in dir(searchAgents):
        heur = getattr(searchAgents, heuristic)
      elif heuristic in dir(search):
        heur = getattr(search, heuristic)
      else:
        raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
      print('[SearchAgent Class] Using Function %s And heuristic %s::' % (fn, heuristic))
      # Note: this bit of Python trickery combines the search algorithm and the heuristic
      self.searchFunction = lambda x: func(x, heuristic=heur)

    # Get the search problem type from the name
    if prob not in dir(searchAgents) or not prob.endswith('Problem'):
      raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
    self.searchType = getattr(searchAgents, prob)
    print('[SearchAgent Class] Using Problem Type:: ' + prob)

  def registerInitialState(self, state):
    if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
    starttime = time.time()
    problem = self.searchType(state) # Makes a new search problem
    self.actions  = self.searchFunction(problem) # Find a path
    totalCost = problem.getCostOfActions(self.actions)
    print('Path Found With Total Cost Of %d In %.1f Seconds::' % (totalCost, time.time() - starttime))
    if '_expanded' in dir(problem): print('Search Nodes Expanded:: %d' % problem._expanded)

  def getAction(self, state):
    if 'actionIndex' not in dir(self): self.actionIndex = 0
    i = self.actionIndex
    self.actionIndex += 1
    if i < len(self.actions):
      return self.actions[i]
    else:
      return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
  def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True):
    self.walls = gameState.getWalls()
    self.startState = gameState.getPacmanPosition()
    print self.walls
    print self.walls.width, self.walls.height, self.startState, goal
    w = self.walls.width -1
    h = self.walls.height -1

    if start != None: self.startState = start
    self.goal = goal
    self.costFn = costFn
    if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
      print 'Warning: This Does Not Look Like A Regular Search Maze'

    # For display purposes
    self._visited, self._visitedlist, self._expanded = {}, [], 0

    print self.getSuccessors(self.startState)
    state_queue= util.Queue();
    visited = set()
    parents = dict()
    direction = dict()
    path = list()
    state_queue.push(self.getStartState())
    parents[self.getStartState()] = (-1,-1);
    direction[self.getStartState()] = "null";
    start_state = self.getStartState()

    # Creating the Maze files by using the BFS based search method
    f = open('maze.P', 'w')
    f1 = open('maze.pl', 'w')
    g = open('mazeastar.P', 'w')
    g.write(":- op(400,yfx,'#').\n")
    visited.add(self.getStartState());
    i=0
    #goal = 0
    "This while loop keep iterating till the queue gets empty or the goal is reached"
    while not state_queue.isEmpty():
      curr_state = state_queue.pop()
      children = self.getSuccessors(curr_state)
      "This for loop iterates over the successor to put the elements in queue and stop once the goal is reached"
      for child in children:
        "print child, child[0]"
        i = i+1
        #if child[0] not in visited:
        if child[0] not in visited:
            state_queue.push(child[0])
            heur = abs( child[0][0] - goal[0] ) + abs( child[0][1] - goal[1] )
            #print "check"
            #print curr_state, child[0], child[1], child[2], (curr_state[0]-1)*h + curr_state[1], (child[0][0]-1)*h + child[0][1]
            if curr_state == start_state:
              f.write("connected(start, cell%d, %s).\n"% ((child[0][0]-1)*h + child[0][1], child[1].lower()))
              f1.write("connected(start, cell%d, %s).\n"% ((child[0][0]-1)*h + child[0][1], child[1].lower()))
              g.write("connected(start#cell%d#%s#%d#%d).\n"% ((child[0][0]-1)*h + child[0][1], child[1].lower(), heur, child[2]))
            else:
              f.write("connected(cell%d, cell%d, %s).\n"% ((curr_state[0]-1)*h + curr_state[1], (child[0][0]-1)*h + child[0][1], child[1].lower()))
              f1.write("connected(cell%d, cell%d, %s).\n"% ((curr_state[0]-1)*h + curr_state[1], (child[0][0]-1)*h + child[0][1], child[1].lower()))
              g.write("connected(cell%d#cell%d#%s#%d#%d).\n"% ((curr_state[0]-1)*h + curr_state[1], (child[0][0]-1)*h + child[0][1], child[1].lower(), heur, child[2]))
      visited.add(curr_state)
    f.write("goal(cell1,null).\n")
    f1.write("goal(cell1,null).\n")
    g.write("goal(cell1).\n")
    print "i=", i


  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
     isGoal = state == self.goal

     # For display purposes only
     if isGoal:
       self._visitedlist.append(state)
       import __main__
       if '_display' in dir(__main__):
         if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
           __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

     return isGoal

  def getSuccessors(self, state):
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextState = (nextx, nexty)
        cost = self.costFn(nextState)
        successors.append( ( nextState, action, cost) )

    # Bookkeeping for display purposes
    self._expanded += 1
    if state not in self._visited:
      self._visited[state] = True
      self._visitedlist.append(state)

    return successors

  def getCostOfActions(self, actions):
    if actions == None: return 999999
    x,y= self.getStartState()
    cost = 0
    for action in actions:
      # Check figure out the next state and see whether its' legal
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]: return 999999
      cost += self.costFn((x,y))
    return cost

class StayEastSearchAgent(SearchAgent):
  def __init__(self):
      self.searchFunction = search.uniformCostSearch
      costFn = lambda pos: .5 ** pos[0]
      self.searchType = lambda state: PositionSearchProblem(state, costFn)

class StayWestSearchAgent(SearchAgent):
  def __init__(self):
      self.searchFunction = search.uniformCostSearch
      costFn = lambda pos: 2 ** pos[0]
      self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
  "The Manhattan distance heuristic for a PositionSearchProblem"
  xy1 = position
  xy2 = problem.goal
  return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
  "The Euclidean distance heuristic for a PositionSearchProblem"
  xy1 = position
  xy2 = problem.goal
  return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

class CornersProblem(search.SearchProblem):
  def __init__(self, startingGameState):

    print "In CornersProblem Class"

    self.walls = startingGameState.getWalls()
    self.startingPosition = startingGameState.getPacmanPosition()
    top, right = self.walls.height-2, self.walls.width-2
    self.corners = ((1,1), (1,top), (right, 1), (right, top))
    for corner in self.corners:
        if not startingGameState.hasFood(*corner):
            print 'Warning: no food in corner ' + str(corner)
    self.costFn = lambda x: 1
    self._expanded = 0 # Number of search nodes expanded

    cornerMask = (0b1000, 0b0100, 0b0010, 0b0001)

  def getStartState(self):
    print "In getStartState()"
    return self.startState


  def isGoalState(self, state):
    print "In isGoalState()"
    isGoal = True
    "It passes true if all the corners have been visited and the tuple 'corners not visited' has no element in it"
    if len(state[1]) !=0:
        isGoal = False

    return isGoal

  def getSuccessors(self, state):
    print "In getSuccessors()"
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:

      "We check if the location of the successor has food. If it has food, its location is removed from not visited list"
      x,y = state[0]
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextState = (nextx, nexty)
        if nextState in state[1]:
            notvisitedList = list(state[1])
            notvisitedList.remove(nextState)
            successors.append( ( (nextState,tuple(notvisitedList)), action, 1) )
        else:
            successors.append( ( (nextState,state[1]), action, 1) )


    self._expanded += 1
    return successors

  def getCostOfActions(self, actions):
    print "In getCostOfActions()"
    if actions == None: return 999999
    x,y= self.startingPosition
    for action in actions:
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]: return 999999
    return len(actions)


def cornersHeuristic(state, problem):
  print
  corners = problem.corners # These are the corner coordinates
  walls = problem.walls # These are the walls of the maze, as a Grid (game.py)
  position = state[0]
  maxi = 0
  for i in range(0,len(state[1])):
    #print state
    dis = abs(position[0]-state[1][i][0]) + abs(position[1]-state[1][i][1])
    if(dis > maxi):
      maxi = dis
  return maxi

class AStarCornersAgent(SearchAgent):
  def __init__(self):
    print "AStarCornersAgent class init()"
    self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
    self.searchType = CornersProblem

class FoodSearchProblem:
  def __init__(self, startingGameState):
    print "FoodSearchProblem class init()"
    self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
    self.walls = startingGameState.getWalls()
    self.startingGameState = startingGameState
    self._expanded = 0
    self.heuristicInfo = {} # A dictionary for the heuristic to store information

  def getStartState(self):
    print "getStartState()"
    return self.start

  def isGoalState(self, state):
    print "isGoalState()"
    return state[1].count() == 0

  def getSuccessors(self, state):
    print "getSuccessors"
    successors = []
    self._expanded += 1
    for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state[0]
      dx, dy = Actions.directionToVector(direction)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextFood = state[1].copy()
        nextFood[nextx][nexty] = False
        successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
    return successors

  def getCostOfActions(self, actions):
    x,y= self.getStartState()[0]
    cost = 0
    for action in actions:
      # figure out the next state and see whether it's legal
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]:
        return 999999
      cost += 1
    return cost

class AStarFoodSearchAgent(SearchAgent):
  "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
  def __init__(self):
    self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
    self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
  position, foodGrid = state

  "The current heuristic we are using denotes the minimum spanning tree for the current position of the node. We start from the nearest food the node has. After that we keep recursively going to the next nearest node until we exhaust the list of food nodes. It expands close to 7100 nodes"
  foodlist = foodGrid.asList()
  foodcoord = foodlist[0]
  mini = float("inf")
  for coord in foodlist:
    tmpdist = abs(position[0]-coord[0]) + abs(position[1]-coord[1])
    if(mini > tmpdist):
      mini = tmpdist
      foodcoord = coord

  foodlist.remove(coord)
  while len(foodlist)>0:
    dist = float("inf")
    foodcoord = []
    for food in foodlist:
      tmpDist = abs(food[0]-coord[0]) + abs(food[1]-coord[1])
      if(dist > tmpDist):
        dist = tmpDist
        coord2 = food
    mini += dist
    coord = coord2
    foodlist.remove(coord)

  return mini

class ClosestDotSearchAgent(SearchAgent):
  def registerInitialState(self, state):
    self.actions = []
    currentState = state
    while(currentState.getFood().count() > 0):
      nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
      self.actions += nextPathSegment
      for action in nextPathSegment:
        legal = currentState.getLegalActions()
        if action not in legal:
          t = (str(action), str(currentState))
          raise Exception, 'findPathToClosestDot Returned An Illegal Move: %s!\n%s' % t
        currentState = currentState.generateSuccessor(0, action)
    self.actionIndex = 0
    print 'Path Found With Cost::%d.' % len(self.actions)

  def findPathToClosestDot(self, gameState):
    startPosition = gameState.getPacmanPosition()
    food = gameState.getFood()
    walls = gameState.getWalls()
    problem = AnyFoodSearchProblem(gameState)

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

class AnyFoodSearchProblem(PositionSearchProblem):
  def __init__(self, gameState):
    self.food = gameState.getFood()
    self.walls = gameState.getWalls()
    self.startState = gameState.getPacmanPosition()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0

  def isGoalState(self, state):
    x,y = state
    util.raiseNotDefined()

class ApproximateSearchAgent(Agent):
  def registerInitialState(self, state):
    "*** YOUR CODE HERE ***"

  def getAction(self, state):
    util.raiseNotDefined()

def mazeDistance(point1, point2, gameState):
  x1, y1 = point1
  x2, y2 = point2
  walls = gameState.getWalls()
  assert not walls[x1][y1], 'point1 is a wall: ' + point1
  assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
  prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False)
  print "Problem::",
  print prob
  print "Calling BFS Function 1::"
  return len(search.breadthFirstSearch1(prob))