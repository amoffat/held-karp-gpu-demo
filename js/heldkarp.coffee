assert = (condition, message) ->
    if !condition
        throw message or "Assertion failed"

# cartesian product
product = (a, b, combine=(elA, elB) -> [elA, elB]) ->
    res = []
    _.each a, (elA) ->
        _.each b, (elB) ->
            res.push(combine elA, elB)
    res

# a pointer-based implementation of combination generator
combinations = (elements, k) ->
    if k == 0
        return [[]]

    lastIdx = elements.length-1
    pointers = [0...k]
    endTest = lastIdx-k

    lastPtrIdx = pointers.length-1

    resetPointers = (pIdx) ->
        prevIdx = pointers[pIdx]
        pIdx++
        _.each _.slice(pointers, pIdx), ->
            newIdx = prevIdx + 1
            pointers[pIdx] = newIdx
            prevIdx = newIdx
            pIdx++

    capture = -> _.map pointers, (idx) -> elements[idx]

    canMove = (pIdx, pointers) ->
        val = pointers[pIdx]
        maxAllowedIdx = elements.length-k+pIdx
        val+1 <= maxAllowedIdx

    pIdx = lastPtrIdx
    results = []
    results.push capture()

    while true
        if canMove pIdx, pointers
            pointers[pIdx] += 1
            if pIdx < lastPtrIdx
                resetPointers pIdx
                pIdx = lastPtrIdx
            results.push capture()

        else if pIdx == 0
            break

        else
            pIdx--

    results


# a recursion-based combination generator.  slower than the pointer based
# version
combinations2 = (elements, k) ->
    if k > elements.length
        combos = []

    else if k == 1
        combos = ([el] for el in elements)

    else
        first = elements[0]
        slice = _.slice elements, 1

        sub1 = combinations2 slice, k-1
        sub1 = product [first], sub1, _.concat

        sub2 = combinations2 slice, k

        combos = _.concat sub1, sub2

    combos


# produce the family of all possible sets from the universe of elements
powerset = (elements) ->
    fn = _.partial combinations, elements
    _.map([0..elements.length], fn)



testCombinations = ->
    results = combinations(["a", "b", "c", "d", "e"], 2)
    correct = [["a", "b"], ["a", "c"], ["a", "d"], ["a", "e"], ["b", "c"],
        ["b", "d"], ["b", "e"], ["c", "d"], ["c", "e"], ["d", "e"]]
    assert _.isEqual(results, correct)

    results = combinations(["a", "b", "c", "d", "e"], 4)
    correct = [["a", "b", "c", "d"], ["a", "b", "c", "e"], ["a", "b", "d", "e"],
        ["a", "c", "d", "e"], ["b", "c", "d", "e"]]
    assert _.isEqual(results, correct)

    results = combinations(["a", "b", "c", "d", "e"], 5)
    correct = [["a", "b", "c", "d" , "e"]]
    assert _.isEqual(results, correct)

    results = combinations(["a", "b", "c", "d", "e"], 1)
    correct = [["a"], ["b"], ["c"], ["d"] , ["e"]]
    assert _.isEqual(results, correct)

    results = combinations(["a", "b", "c"], 0)
    correct = [[]]
    assert _.isEqual(results, correct)

    results = combinations([1,2,3,4,5], 3)
    correct = [[1,2,3], [1,2,4], [1,2,5], [1,3,4], [1,3,5], [1,4,5], [2,3,4],
        [2,3,5], [2,4,5], [3,4,5]]
    assert _.isEqual(results, correct)


testProduct = ->
    results = product([], [])
    correct = []
    assert _.isEqual(results, correct)

    results = product(["a", "b"], ["c"])
    correct = [["a", "c"], ["b", "c"]]
    assert _.isEqual(results, correct)

    results = product(["c"], ["a", "b"])
    correct = [["c", "a"], ["c", "b"]]
    assert _.isEqual(results, correct)

    results = product([], ["a", "b"])
    correct = []
    assert _.isEqual(results, correct)


runTests = ->
    testCombinations()
    testProduct()

@generateLevels = (nodes) ->
    nodesMinusStart = _.without(_.map(_.keys(nodes), _.toInteger), 0)
    family = powerset(nodesMinusStart)

    allLevels = []
    for subset in family
        levels = []
        for goThrough in subset
            endAt = _.difference(nodesMinusStart, goThrough)
            if _.isEmpty endAt
                endAt = [0]
            level = product(endAt, [goThrough])
            levels = _.concat levels, level

        allLevels.push levels

    allLevels


createKey = (list) ->
    _.join list, ","

findCost = (endAt, goThrough, cache, nodes) ->

    bestCost = Infinity
    bestParent = null

    if _.isEqual goThrough, []
        bestParent = 0
        bestCost = nodes[0][endAt]

    else
        combos = combinations goThrough, goThrough.length-1

        for combo in combos
            potentialParent = (_.difference goThrough, combo)[0]
            key = createKey _.concat(potentialParent, combo)
            [subCost, subParent] = cache[key]
            totalCost = subCost + nodes[potentialParent][endAt]

            if totalCost < bestCost
                bestCost = totalCost
                bestParent = potentialParent

    [bestCost, bestParent]


backTrackParents = (cache, levels, endParent) ->
    path = [endParent]

    lastLevel = _.last levels
    subPath = lastLevel[0][1]

    while not _.isEmpty subPath
        subPath = _.difference subPath, [endParent]
        key = createKey _.concat(endParent, subPath)
        endParent = cache[key][1]

        path.push endParent

    path


@heldKarp = (nodes, levels) ->
    cache = {}

    for level, levelNum in levels
        for subset in level
            [endAt, goThrough] = subset
            [cost, parent] = findCost endAt, goThrough, cache, nodes
            key = createKey _.concat(endAt, goThrough)
            cache[key] = [cost, parent]

    path = backTrackParents cache, levels, parent
    path.reverse()
    path.push 0
    path


@generateTour = (num, maxEdgeCost=100) ->
    nodes = {}
    getCost = -> _.toInteger Math.random() * maxEdgeCost

    for node in [0...num]
        edges = {}
        nodes[node] = edges

        for edge in [0...num]
            cost = getCost()
            edges[edge] = cost

    nodes


@distance = (a, b) ->
    x = b.x-a.x
    y = b.y-a.y
    Math.sqrt x*x+y*y


createNode = (id) ->
    x = Math.random()
    y = Math.random()
    node = {
        id: id
        pos:
            x: x
            y: y
    }

@computeEdges = (nodes) ->
    for key, n1 of nodes
        n1.edges = {}
        for i, n2 of nodes
            d = distance n1.pos, n2.pos
            n1.edges[i] = d


@pushNode = (nodes) ->
    id = _.size nodes
    nodes[id] = createNode id
    @computeEdges nodes

@popNode = (nodes) ->
    id = (_.size nodes) - 1
    delete nodes[id]
    @computeEdges nodes


@generateNodes = (num, asymmetric=false) ->
    nodes = {}
    for i in [0...num]
        nodes[i] = createNode i

    @computeEdges nodes
    nodes
