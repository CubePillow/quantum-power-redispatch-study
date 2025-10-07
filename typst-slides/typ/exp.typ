#import "/globals.typ": *

= Experiments
== Effect of normalisation
#slide(composer: (2.5fr, 3fr), gutter: 1em)[
    #set text(size: 20pt)
    - Because we know the minimum and maximum values of our
      objective functions
      - Product Cost and Number of Overloads minimized when all
        generators shut down
      - Switching Cost minimized when no switches occurs
    - We can normalise our objective functions before finding weights
      for the multi-objective function
][
    #set text(size: 18pt)
    #set align(center + bottom) 
    #image("../img/norm.png", height: 45%)
    We used $lambda_P = 1$ for the\ normalised multi-objective
    QUBO and\ $lambda_P = 1000$ for the un-normalised QUBO
]
== Weights on larger problems
#slide(composer: (2.5fr, 3fr), gutter: 1em)[
    #set text(size: 20pt)
    - Because the process of finding these weights
      is actually quite expensive
      - Requires solving the multi-objective problem at each iteration
    - We want to apply weights on smaller problems to
      redispatch problems with more timesteps
    - Able to maintain $< 0.03%$ power production deviation
][
    #set text(size: 18pt)
    // #set align(center + bottom) 
    #set align(center + horizon)
    #muchpdf(read("../img/apply-norm.pdf", encoding: none), height:80%)
    Results from applying weights from the $T=2$ redispatch problem to larger problems.
]
== Redispatch on quantum hardware
#set align(horizon)
#slide(composer: (2.5fr, 3fr), gutter: 1em)[
    #set text(size: 20pt)
    - We used D-Wave's Advantage System 4.1 quantum annealer 
    - Quantum annealer generally has worse performance compared to classical solver
    - Aligns with understanding that current day quantum computers that are still in its infancy.
][
    #set text(size: 18pt)
    // #set align(center + bottom) 
    #set align(center + horizon)
    #muchpdf(read("../img/compare-solvers.pdf", encoding: none), height:80%)
]
