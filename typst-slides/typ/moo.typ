#import "/globals.typ": *

// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))

= Finding Weights for our Multi-Objective QUBO
== A Posteriori Approach using Averaging Adaptive Method
#slide(composer: (2.5fr, 3fr), repeat: 2)[
    #v(1em)
    #align(horizon)[
        - find a set of possible weights
          (using Averaging Adaptive Method)
        - choose "best" weight based on some user defined criteria (
          $<0.1%$ power production deviation, sorted by
          $Q_O$, $Q_C$, $Q_S$)
    ]
][
    #set align(center + horizon)
    #cetz-canvas({
        import cetz.draw: *

        content((0,0),
            muchpdf(read("../img/pf.pdf", encoding: none)))
        (pause, )
        circle((-0.3, -0.5), radius: 0.5, stroke: red)
    })
]
