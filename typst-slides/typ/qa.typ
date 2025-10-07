#import "/globals.typ": *


= Quantum Annealing
== Adiabatic Theorem
#slide()[
    #let plot-w = 140pt
    #let plot-h = 70pt
    #align(center + horizon)[
        Key to Quantum Annealing is the Adiabatic Theorem\
    #set text(size: 22pt)
        where for "slow enough" pertubations from $H_0 arrow H_1$
        #set text(size: 19pt)
        
    #lq.diagram(
        width: plot-w, height: plot-h,
        lq.plot(range(3), (2.5, 3, 2.5), smooth: true, color: black),
        lq.place(0.25, 2.7, circle(fill: red, height: 0.7em)),
        lq.plot(range(3), (3.6, 3.3, 3.6), smooth: true, color: black),
        lq.plot(range(3), (4, 3.7, 4), smooth: true, color: black),
        lq.plot(range(3), (4.5, 4, 4.5), smooth: true, color: black),
        title: [$H_0$],
        xlabel: "t", ylabel: "energy", grid: none,
        xaxis: (ticks: none), yaxis: (ticks: none)
    )
        #h(3em)
    #lq.diagram(
        width: plot-w, height: plot-h,
        lq.plot(range(3), (2.5, 3, 2.5), smooth: true, color: black),
        lq.place(1, 3, circle(fill: red, height: 0.7em)),
        lq.plot(range(3), (3.6, 3.3, 3.6), smooth: true, color: black),
        lq.plot(range(3), (4, 3.7, 4), smooth: true, color: black),
        lq.plot(range(3), (4.5, 4, 4.5), smooth: true, color: black),
        xlabel: "t", ylabel: "energy", grid: none,
        xaxis: (ticks: none), yaxis: (ticks: none),
        title: [$a(t)H_0 + b(t)H_1$],
    )
        #h(3em)
    #lq.diagram(
        width: plot-w, height: plot-h,
        lq.plot(range(3), (2.5, 3, 2.5), smooth: true, color: black),
        lq.place(1.75, 2.7, circle(fill: red, height: 0.7em)),
        lq.plot(range(3), (3.6, 3.3, 3.6), smooth: true, color: black),
        lq.plot(range(3), (4, 3.7, 4), smooth: true, color: black),
        lq.plot(range(3), (4.5, 4, 4.5), smooth: true, color: black),
        xlabel: "t", ylabel: "energy", grid: none,
        xaxis: (ticks: none), yaxis: (ticks: none),
        title: [$H_1$],
    )
        
        #set text(size: 20pt)
         $H_0$ is a Hamiltonian with a known ground state\
        and $H_1$ is a Hamiltonian whose ground state we want to find.
    ]
]
== Quadratic Unconstrained Binary Optimisation
#slide(composer: (1fr, 3fr), gutter: 0em)[
    #set align(center + horizon)
    #set text(size: 0.7em)
    #cetz.canvas({
        import cetz.draw: *
        let ymax = 7
        let xmax = 4
        let icon-h = 30pt
        
        circle((0, 3), radius: (3, 2.5))
        circle((0, 0), radius: (3, 2.5))
        circle((0, -1.5), radius: (2, 1))

        content((0,-1.5), "P")
        content((0,0), "NP")
        content((0,1.5), "NP-Complete")
        content((0,3.5), "NP-Hard")
        content((-2,2.7), text(size: 0.65em, "QUBO", weight: "black"))
    })
    QUBO is NP-Hard in General
][
    #set align(center)
    #set text(size: 18pt)
    QUBO problems have the form:
    $
        min_(bold(x) in {0,1}^n) bold(x)^top bold(Q) bold(x) =
        min_bold(x) vec(x_1, dots.v, x_n)^top
        mat(
        Q_(1,1), ..., Q_(1,n);
        dots.v, dots.down, dots.v;
        Q_(n,1), ..., Q_(n,n);
        )
        vec(x_1, dots.v, x_n)
    $
    $bold(Q)$ is equivalent to a Hamiltonian
    up to a change of variable.
    
    #set text(size: 15pt)
    #let plot-w = 120pt
    #let plot-h = 60pt
    #lq.diagram(
        width: plot-w, height: plot-h,
        lq.plot(range(3), (2.5, 3, 2.5), smooth: true, color: black),
        lq.place(0.25, 2.7, circle(fill: red, height: 0.7em)),
        lq.plot(range(3), (3.6, 3.3, 3.6), smooth: true, color: black),
        lq.plot(range(3), (4, 3.7, 4), smooth: true, color: black),
        lq.plot(range(3), (4.5, 4, 4.5), smooth: true, color: black),
        title: [$H_0$],
        xlabel: "t", ylabel: "energy", grid: none,
        xaxis: (ticks: none), yaxis: (ticks: none)
    )
    #lq.diagram(
        width: plot-w, height: plot-h,
        lq.plot(range(3), (2.5, 3, 2.5), smooth: true, color: black),
        lq.place(1, 3, circle(fill: red, height: 0.7em)),
        lq.plot(range(3), (3.6, 3.3, 3.6), smooth: true, color: black),
        lq.plot(range(3), (4, 3.7, 4), smooth: true, color: black),
        lq.plot(range(3), (4.5, 4, 4.5), smooth: true, color: black),
        xlabel: "t", ylabel: "energy", grid: none,
        xaxis: (ticks: none), yaxis: (ticks: none),
        title: [$a(t)H_0 + b(t)H_1$],
    )
    #lq.diagram(
        width: plot-w, height: plot-h,
        lq.plot(range(3), (2.5, 3, 2.5), smooth: true, color: black),
        lq.place(1.75, 2.7, circle(fill: red, height: 0.7em)),
        lq.plot(range(3), (3.6, 3.3, 3.6), smooth: true, color: black),
        lq.plot(range(3), (4, 3.7, 4), smooth: true, color: black),
        lq.plot(range(3), (4.5, 4, 4.5), smooth: true, color: black),
        xlabel: "t", ylabel: "energy", grid: none,
        xaxis: (ticks: none), yaxis: (ticks: none),
        title: [$H_1$],
    )
    
    #set text(size: 18pt)
    Find solution to $min_bold(x) bold(x)^top bold(Q) bold(x)$
    via quantum annealing.
]

