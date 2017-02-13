TOUR_COLOR = "green"
NODE_PADDING = [100, 20]
MAX_NODES = 16
DEFAULT_NUM_NODES = 13
MIN_NODES = 3
MAX_PERFORMANCE_POINTS = 10

# for benchmarking
if not _.isUndefined performance
    now = -> performance.now()
else
    now = -> Date.now()


chart = null
chartDef = $.Deferred()
chartPromise = chartDef.promise()


# because our for-loops must only contain hard-coded values (the gpu compiler
# unrolls for-loops), we must rebuild our kernel and preamble every time we add
# or remove a node from the TSP
constructComputationKernels = (numNodes) ->

    preamble = """
    uniform sampler2D level_tex;
    uniform sampler2D subpath_tex;
    uniform vec2 level_tex_size;
    uniform float cost_matrix[256];
    uniform int num_nodes;
    uniform int cur_level;

    /*
     * performs our cost matrix lookup, from node a to node b
     */
    float cost_between(int a, int b) {
        int idx = num_nodes * a + b;
        float cost = 0.0;

        // the uglist array lookup in the world
        for (int i=0; i<256; i++) {
            if (i == idx) {
                cost = cost_matrix[i];
                break;
            }
        }
        return cost;
    }

    /*
     * takes a num and a 0-based bit_idx and returns if the bit was on
     * at that index.  if it was on, flip it off, and set the resulting
     * value to bit_off
     */
    bool bit_on(int num, int bit_idx, out int bit_off) {
        int shift = int(pow(2.0, float(bit_idx)));
        int tmp = num / shift;
        bool bit_on = int(mod(float(tmp), 2.0)) != 0;

        bit_off = num;
        if (bit_on) {
            bit_off = num - int(pow(2.0, float(bit_idx)));
        }
        return bit_on;
    }

    /*
     * takes a pixel's worth of subpath level query data and hashes
     * it to an index
     */
    int hash_pixel(in ivec3 pixel) {
        int idx = pixel.r*65536 + pixel.g*256 + pixel.b;
        return idx;
    }
    """

    kernel = """
    /*
     * assignment to src (which will be 0) initially prevents the gpu compiler from
     * optimizing away src, and related, tex_inputs
     */
    float min_cost = src.r;
    int parent;

    vec4 level_data = texture2D(level_tex, texcoord);
    int checking_level = 255-int(level_data.a*255.0);
    int end_at = int(level_data.r*255.0);

    int go_through = int(level_data.g*255.0);
    int go_through2 = int(level_data.b*255.0);
    int cur_go_through = go_through;

    const int num_nodes_real = #{numNodes};


    /*
     * since our shader runs on every subpath query on every level, simultaneously,
     * we need a way to short-circuit queries that it's not time to solve yet.
     */
    if (checking_level != cur_level) {
        discard;
        return;
    }

    if (cur_level == 0) {
        min_cost = cost_between(0, end_at);
        parent = 0;
    }
    else {
        for (int check_parent=0; check_parent<num_nodes_real; check_parent++) {

            bool second_byte = check_parent >= 8;

            int check_adjusted = check_parent;
            if (second_byte) {
                cur_go_through = go_through2;
                check_adjusted -= 8;
            }

            int bit_off;
            bool was_on = bit_on(cur_go_through, check_adjusted, bit_off);

            if (was_on) {
                int subpath_idx;
                ivec3 pixel;
                pixel.r = check_parent;

                if (second_byte) {
                    pixel.g = go_through;
                    pixel.b = bit_off;
                }
                else {
                    pixel.g = bit_off;
                    pixel.b = go_through2;
                }

                subpath_idx = hash_pixel(pixel);

                vec2 subpath_texcoord;
                subpath_texcoord.x = mod(float(subpath_idx), level_tex_size.x) / level_tex_size.x;
                subpath_texcoord.y = (float(subpath_idx) / level_tex_size.x) / level_tex_size.y;

                vec4 subpath = texture2D(subpath_tex, subpath_texcoord);
                float cost = subpath.x + cost_between(check_parent, end_at);
                if (cost < min_cost) {
                    min_cost = cost;
                    parent = check_parent;
                }

            }
        }

    }


    dst.x = min_cost;
    dst.y = float(parent);
    dst.z = float(cur_level);
    dst.w = float(cur_go_through);
    """

    [preamble, kernel]




benchmark = (fn) ->
    start = now()
    fn()
    now() - start

toggleSymmetric = (cy, symmetric) ->
    curveStyle = "bezier"
    opacity = 0.1
    if symmetric
        curveStyle = "haystack"
        opacity *= 0.5

    style = cy.style()
    style.selector("edge.route").style(
        "curve-style": curveStyle
        opacity: opacity
    ).update()

    cy.style().selector("edge.tour").style(
        "curve-style": "haystack"
        opacity: 1
    ).update()


# converts a normalized 0-1 space value to a graph pixel value, taking into
# account the graph's padding
normToGraph = (cy, val) ->
    graphWidth = cy.width()
    graphHeight = cy.height()
    newWidth = val[0] * (graphWidth - NODE_PADDING[0]*2) + NODE_PADDING[0]
    newHeight = val[1] * (graphHeight - NODE_PADDING[1]*2) + NODE_PADDING[1]
    [newWidth, newHeight]

# converts a graph pixel space value to a normalized 0-1 value, taking into
# account the graph's padding
graphToNorm = (cy, val) ->
    graphWidth = cy.width()
    graphHeight = cy.height()

    newWidth = (val[0] - NODE_PADDING[0]) / (graphWidth - NODE_PADDING[0]*2)
    newHeight = (val[1] - NODE_PADDING[1]) / (graphHeight - NODE_PADDING[1]*2)
    [newWidth, newHeight]


graphNodes = (cy, solver, nodes, symmetric=true) ->
    toggleSymmetric cy, symmetric

    for i, node of nodes
        pos = node.pos
        newPos = normToGraph cy, [pos.x, pos.y]
        cy.add {
            data:
                id: node.id
            selectable: false
            position:
                x: newPos[0]
                y: newPos[1]
        }

    #get = cy.getElementById
    get = (id) -> cy.$ "##{id}"

    for src, node of nodes
        for target of node.edges
            if src == target
                continue

            id = "#{src}-#{target}"

            srcPos = get(src).position()
            targetPos = get(target).position()

            cost = distance srcPos, targetPos

            cy.add {
                data:
                    id: id
                    cost: cost
                    source: src
                    target: target
                classes: "route"
                selectable: false
            }

    recomputeTSP cy, solver


getNumNodes = (cy) ->
    cy.nodes().length

# extracts nodes from the cy graph to be augmented and fed back into the cy
# graph
extractNodesForAugment = (cy) ->
    nodes = {}

    cy.nodes().forEach (cyNode) ->
        id = cyNode.id()
        newPos = graphToNorm cy, [cyNode.position("x"), cyNode.position("y")]
        node = {
            id: parseInt id
            pos:
                x: newPos[0]
                y: newPos[1]
        }
        nodes[id] = node

    computeEdges nodes
    nodes

# extract node data from the cy graph to be used in a solver
extractNodesForSolver = (cy) ->
    nodes = {}

    cy.edges().forEach (edge) ->
        src = edge.source().data "id"
        target = edge.target().data "id"
        cost = edge.data "cost"

        targets = nodes[src]
        if _.isUndefined targets
            targets = {}
            nodes[src] = targets

        targets[target] = cost

    nodes


edgesFromPath = (path, fn) ->
    indices = [0..path.length]

    for i in _.slice(indices, 0, indices.length-2)
        src = path[i]
        target = path[i+1]
        fn src, target


colorizeTSP = (cy, path) ->
    cy.edges().removeClass "tour"

    edgesFromPath path, (src, target) ->
        edgeId = "#{src}-#{target}"
        edge = cy.$("##{edgeId}")
        edge.addClass "tour"

        node = cy.getElementById(src)
        node.addClass "tour"


# a helper to run a function asynchronously and return a promise
async = (fn) ->
    def = $.Deferred()
    _.defer ->
        def.resolve fn()
    def.promise()


getCost = (nodes, path) ->
    cost = 0
    lastIdx = path[0]

    for idx in _.slice(path, 1)
        cost += nodes[lastIdx][idx]
        lastIdx = idx
    cost

recomputeTSP = (cy, solver) ->
    nodes = extractNodesForSolver cy

    pathPromise = solver nodes
    pathPromise.then (data) ->
        [path, elapsed] = data

        cost = getCost nodes, path
        updateUICost cost, nodes, solver
        updateUITime elapsed, nodes, solver

        colorizeTSP cy, path


# triggered on a move node event.  we must update the cost of all the edges
# connected to this node
updateCost = (ev) ->
    cy = ev.cy
    node = ev.cyTarget

    edges = node.connectedEdges()

    edges.forEach (edge) ->
        a = node
        b = edge.target()

        if b == a
            b = node
            a = edge.source()

        dist = distance a.position(), b.position()
        edge.data "cost", dist


# load an image from a url or path, return a promise
loadImage = (src) ->
    def = $.Deferred()
    img = new Image()

    img.onload = -> def.resolve img
    img.error = def.reject

    img.src = src

    def.promise()


parsePixels = (pixels) ->
    goThrough = []

    idx = 0
    for i in [1..16]
        if i % 8 == 0
            idx++

        mask = 1 << (i % 8)
        val = pixels[idx]

        if val & mask
            goThrough.push i

    goThrough


unpackImageData = (data) ->
    entries = []
    for r, i in data by 4
        g = data[i+1]
        b = data[i+2]

        endAt = r
        goThrough = parsePixels [g, b]
        entries.push [endAt, goThrough]

    entries


unpackImage = (img) ->
    canvas = $("#unpack-canvas")[0]
    canvas.width = img.width
    canvas.height = img.height

    ctx = canvas.getContext "2d"
    ctx.drawImage img, 0, 0, img.width, img.height
    data = (ctx.getImageData 0, 0, img.width, img.height).data

    unpackImageData data


# loads a set of images associated with a level
# FOR THE CPU SOLVER ONLY
loadSubpathLevels = (nodeCount) ->
    promises = []

    for i in [0...nodeCount]
        src = "levels/#{nodeCount}/#{i}.png"
        promise = loadImage src
        promises.push promise

    promise = $.when.apply $, promises
    promise.then (stuff...) ->
        stuff


costMatrixFromNodes = (nodes) ->
    size = _.size nodes
    costs = []

    for i in [0...size]
        for j in [0...size]
            cost = nodes[i][j]
            if _.isUndefined cost
                cost = 0
            costs.push cost

    costs


buildCPUSolver = ->
    # the original level generator, doing it at runtime
    levelGen = (nodes) ->
        async -> generateLevels(nodes)

    # the level generator that loads levels from the server in the form of
    # images
    levelGen = (nodes) ->
        numNodes = _.size nodes
        promise = loadSubpathLevels numNodes
        promise.then (images) ->
            levels = _.map images, unpackImage

    levelGen = _.memoize levelGen, (nodes) -> _.size nodes

    (nodes) ->
        levelPromise = levelGen nodes
        levelPromise.then (levels) ->
            start = now()
            path = heldKarp nodes, levels
            elapsed = now() - start
            [path, elapsed]


# converts an RGB pixel value into an index into our giant array of heldKarp
# dynamic programming subproblems
hashPixel = (pixel) ->
    [r,g,b] = pixel
    b1 = r << 16
    b2 = g << 8
    b3 = b
    b1 | b2 | b3

# converts a heldKarp entry -- consisting of a node to end at, and a list of
# nodes to go through -- to an RGB pixel value
packPixel = (endAt, goThrough) ->
    agg = [endAt, 0, 0]

    for num in goThrough
        aggIdx = Math.floor(num / 8) + 1
        val = 1 << (num % 8)
        agg[aggIdx] |= val

    agg


textureLookupCost = (arr, endAt, goThrough) ->
    pixel = packPixel endAt, goThrough
    idx = hashPixel(pixel) * 4

    cost = arr[idx]
    parent = arr[idx+1]
    [cost, parent]


backtrackParents = (nodes, outputs) ->
    num = _.size nodes

    goThrough = [1...num]
    endAt = 0
    outIdx = 1

    path = [endAt]
    while true
        [cost, parent] = textureLookupCost outputs[outIdx], endAt, goThrough
        goThrough = _.difference goThrough, [parent]
        endAt = parent

        path.push parent

        outIdx = (outIdx+1) % 2
        if parent == 0
            break


    path


buildGPUSolver = (engine, numNodes) ->
    levelGen = ->
        src = "levels/#{MAX_NODES}.png"
        promise = loadImage src
        promise.then (image) ->
            storage = Storage.fromImage engine, image, "UNSIGNED_BYTE"
            storage.tex

    size = [4096, 256]
    startingMinCost = 999999
    dummyData = new Float32Array(startingMinCost for i in [0...size[0]*size[1]*4])
    dummyInputs = Storage.fromSize engine, size, dummyData
    output1 = Storage.fromSize engine, size, dummyData
    output2 = Storage.fromSize engine, size, dummyData
    outputs = [output1, output2]

    [preamble, kernel] = constructComputationKernels numNodes

    comp = engine.createComputation size, kernel, preamble, 4

    levelPromise = levelGen()

    (nodes, perf) ->
        num = _.size nodes

        uniforms =
            num_nodes: ["1i", num]
            level_tex_size: ["2fv", size]
            cost_matrix: ["1fv", costMatrixFromNodes(nodes)]

        levelPromise.then (levelTex) ->

            uniforms.level_tex = ["s", levelTex]

            start = now()
            for level in [0...num]
                curOutput = outputs[0]

                uniforms.cur_level = ["1i", level]
                uniforms.subpath_tex = ["s", outputs[1]]

                comp.step dummyInputs, uniforms, curOutput

                # ping-pong!
                outputs.reverse()

            evenOutput = outputs[0].download()
            oddOutput = outputs[1].download()

            path = backtrackParents nodes, [evenOutput, oddOutput]
            elapsed = now() - start
            [path, elapsed]



failGPU = (msg="Unknown") ->
    $("#gpu-fail").show()
    $("#gpu-fail-reason").html(msg)
    $("#gpu-solver-opt").attr("disabled", "disabled")

    chartPromise.then (chart) ->
        data = chart.getDataTable()
        for idx in [MIN_NODES..MAX_NODES]
            data.setValue idx-3, 2, 100



root = $("#gpgpu")
engine = null
try
    engine = getEngine root
catch error
    failGPU error


cpuSolver = buildCPUSolver()
gpuSolver = null
if engine
    gpuSolver = buildGPUSolver(engine, DEFAULT_NUM_NODES)


getSolverType = ->
    $("#selected-solver option:checked").val()

getSolver = ->
    if getSolverType() == "cpu" then cpuSolver else gpuSolver

@cy = cytoscape
    container: $ "#cy"

    userZoomingEnabled: false
    userPanningEnabled: false
    boxSelectionEnabled: false

    style: [
        {
            selector: "node"
            style:
                content: "data(id)"
                "text-valign": "center"
                "color": "white"
                "text-outline-width": 2
                "background-color": "#999"
                "text-outline-color": "#999"
        }

        {
            selector: "edge.route"
            style:
                width: 1.5
                "line-color": "#000"
                opacity: 0.1
                "z-index": 10
        }

        {
            selector: "node.tour"
            style:
                "background-color": TOUR_COLOR
                "text-outline-color": TOUR_COLOR
        }

        {
            selector: "edge.tour"
            style:
                "width": "4px"
                opacity: 1
                "line-color": TOUR_COLOR
                "curve-style": "haystack"
                #"mid-target-arrow-shape": "triangle",
                "mid-target-arrow-color": TOUR_COLOR,
                "z-index": 20
        }

        {
            selector: ".autorotate"
            style:
                "edge-text-rotation": "autorotate"
        }
    ]


updateGraph = (cy, nodes) ->
    graphNodes cy, getSolver(), nodes, true

clearGraph = (cy) -> cy.elements().remove()


# rebuilds the graph with numNodes random nodes.  if numNodes is null, we use
# the global value DEFAULT_NUM_NODES
resetGraph = (cy, numNodes=null) ->
    if numNodes == null
        numNodes = getNumNodes cy

    clearGraph cy
    nodes = generateNodes numNodes
    updateGraph cy, nodes

refreshNodes = (cy, nodes) ->
    if engine and getSolverType() == "gpu"
        gpuSolver = buildGPUSolver engine, _.size nodes
    clearGraph cy
    updateGraph cy, nodes

addNode = (cy) ->
    nodes = extractNodesForAugment cy
    pushNode nodes
    refreshNodes cy, nodes

removeNode = (cy) ->
    nodes = extractNodesForAugment cy
    popNode nodes
    refreshNodes cy, nodes




drawChart = ->
    data = new google.visualization.DataTable()
    data.addColumn "number", "X"
    data.addColumn "number", "CPU"
    data.addColumn "number", "GPU"
    data.addRows(([i, null, null] for i in [MIN_NODES..MAX_NODES]))

    opts =
        chartType: "LineChart"
        containerId: "tsp-perf-chart"
        dataTable: data
        options:
            title: "Logarithmic Performance"
            curveType: "function"
            legend: {position: "right"}
            hAxis:
                title: "Number of nodes"
                ticks: [MIN_NODES..MAX_NODES]
            vAxis:
                title: "Time (ms)"
                logScale: true

    chart = new google.visualization.ChartWrapper opts
    chart._performanceAggs = ([[], []] for i in [MIN_NODES..MAX_NODES])
    chart.draw()
    chartDef.resolve chart


google.charts.load "current", {packages: ["corechart"]}
google.charts.setOnLoadCallback drawChart

chartPromise.then ->
    resetGraph cy, DEFAULT_NUM_NODES

updateUICost = (cost) ->
    $("#tour-cost").text(_.round cost, 2)

updateUITime = (elapsed, nodes, solver) ->
    chartPromise.then (chart) ->
        numNodes = _.size nodes
        idx = numNodes - 3
        solverIdx = if getSolverType() == "cpu" then 1 else 2

        data = chart.getDataTable()

        old = chart._performanceAggs[idx][solverIdx-1]
        if old.length >= MAX_PERFORMANCE_POINTS
            old.shift()
        old.push elapsed
        meanElapsed = _.mean old

        data.setValue idx, solverIdx, meanElapsed

        chart.draw()

        $("#tour-time").text(_.round elapsed)



cy.on "free", (ev) ->
    cy.edges().removeClass "tour"
    cy.nodes().removeClass "tour"

    updateCost ev
    recomputeTSP cy, getSolver()



$("#selected-solver").change ->
    nodes = extractNodesForAugment cy
    refreshNodes cy, nodes


$("#randomize-graph").click -> resetGraph cy
$("#add-node").click (ev) ->
    addNode cy

    num = getNumNodes cy
    $("#remove-node").prop "disabled", false
    if num >= MAX_NODES
        $(ev.target).prop("disabled", true)


$("#remove-node").click (ev) ->
    removeNode cy

    num = getNumNodes cy
    $("#add-node").prop "disabled", false
    if num <= MIN_NODES
        $(ev.target).prop "disabled", true


# render out our latex
$(".math").each (i, el) ->
    katex.render $(el).text(), el

# syntax highlight our code blocks
$ ->
    $("pre code").each (i, block) ->
        hljs.highlightBlock block
