#import "/globals.typ": *

= Power System Redispatch
== Problem Introduction
#slide(repeat: 2, composer: (1fr, 2.1fr),  gutter: 1em, self => [
    #set align(horizon)
    #set text(size: 15pt)
    #only(1   )[
    #figure(
        muchpdf(read("../img/power_network.pdf", encoding: none)),
        numbering: none, gap: 0.2em,
        caption: "Power network in Germany",
    )]
    #only("2")[
        - Ensure:
            - Power produced meet targets ($plus.minus 0.05%$)
            - Controllable Resources only switch to adjacent
              states 
        - Minimise:
            - Cost of production
            - Cost of redispatch orders (state switching)
            - Number of overloaded lines
    ]
],self=>[
    #set align(center)
    #v(1em)
    #let (uncover, only) = utils.methods(self)
    #cetz.canvas({
        import cetz.draw: *
        let ymax = 7.5
        let xmax = 13
        let icon-h = 60pt
        let level-h = icon-h / 3

        content(
            (0, 0), image("../img/wind.svg", height: icon-h),
            name: "wind"
        )
        content(
            (xmax, ymax), image("../img/nuclear.svg", height: icon-h),
            name: "nuc"
        )
        content(
            (0, ymax), image("../img/solar.svg", height: icon-h),
            name: "sol"
        )
        content(
            (xmax, 0), image("../img/coal.svg", height: icon-h),
            name: "coal"
        )
        
        content(
            (3, ymax / 2), image("../img/city.svg", height: icon-h),
            name: "city1"
        )
        content(
            (xmax - 3, ymax / 2), image("../img/city.svg", height: icon-h),
            name: "city2"
        )

        line("sol.south", "city1.north-west", stroke: 4pt)
        line("wind.north", "city1.south-west", stroke: 4pt)
        line("nuc.south", "city2.north-east", stroke: 4pt)
        line("coal.north", "city2.south-east", stroke: 4pt)

        line("city1.east", "city2.west", stroke: 4pt, name: "line")
        only(1, 
            content(
                "line.mid",
                image("../img/guage-full.svg", height: icon-h/1.5),
                anchor: "south",
                padding: 0.3
            )
        )
        only("2-", 
            content(
                "line.mid",
                image("../img/guage-empty.svg", height: icon-h/1.5),
                anchor: "south",
                padding: 0.3
            )
        )
        let add_levels(coord, m, e, add_m: 0, rev: 1) = {
            content(
                (rel: (rev * 0.75, 12pt), to: coord),
                image("../img/money.svg", height: level-h)
            )
            for i in range(1, m) {
                content(
                    (rel: (rev * 0.5, 0)),
                    image("../img/money.svg", height: level-h)
                )
            }
            if add_m > 0 {
                content((rel: (rev * 0.5, 0)), "+")
                for i in range(add_m) {
                    content(
                        (rel: (rev * 0.5, 0)),
                        image("../img/money.svg", height: level-h)
                    )
                }
            }
            content(
                (rel: (rev * 0.75, -18pt), to: coord),
                image("../img/energy.svg", height: level-h)
            )
            for _ in range(e - 1) {
                content(
                    (rel: (rev * 0.5, 0)),
                    image("../img/energy.svg", height: level-h))
            }
        }
        only(
            "1", add_levels("sol.east", 2, 4, rev: 1)
        )
        only(
            "1", add_levels("wind.east", 2, 4, rev: 1)
        )
        only(
            "1", add_levels("nuc.west", 2, 3, rev: -1)
        )
        only(
            "1", add_levels("coal.west", 1, 1, rev: -1)
        )
        only(
            "2-",
            add_levels("sol.east", 2, 4, rev: 1)
        )
        only(
            "2-",
            add_levels("wind.east", 1, 2, add_m: 2, rev: 1)
        )
        only(
            "2-",
            add_levels("nuc.west", 2, 3, rev: -1)
        )
        only(
            "2-",
            add_levels("coal.west", 3, 3, add_m: 1, rev: -1)
        )
    })
])


== Formalised Optimisation Problem
#slide(composer: (1.5fr, 3fr),  gutter: 2em)[
    #v(0.3em)
    #align(center + bottom)[
        #include "ex-prob.typ"
    #let rng = gen-rng-f(5)
    #let T = 50
    #let ts = (0, )
    #let j = 0    
    #for i in random-f(rng, size: T - 1).at(1) {
        // let d = ts.at(0) - ts.at(ts.len() - 1)
        let d = calc.sin(0.5*j) - ts.at(ts.len() - 1)
        let s = if (random-f(rng).at(1)-0.5) < d {1} else {-1}
        ts.push(ts.at(ts.len() - 1) + (i * s))
        j += 1
    }
        
    #set text(size: 18pt)
    Assume we are given $bold(tau)_t$ for $T$ timesteps in the
    future:
        
    #lq.diagram(
        width: 230pt, height: 50pt,
        lq.plot(range(T), ts, smooth: false, color: black),
        title: text([Target Power Production $tau_t$], size:16pt),
        xlabel: lq.label(text([$t$], size: 16pt), pad: 5pt),
        grid: none,
        ylim: (calc.min(..ts) - 0.5, calc.max(..ts) + 0.5),
        xaxis: (ticks: none), yaxis: (ticks: none),
    )
    #set text(size: 14pt)
    ]
][
    #set align(horizon)
    #let coal = box(image("../img/coal.svg"), width: 1em, baseline: 15%)
    #let wind = box(image("../img/wind.svg"), width: 1em, baseline: 15%)
    #let sol = box(image("../img/solar.svg"), width: 1em, baseline: 15%)
    #let nuc = box(image("../img/nuclear.svg"), width: 1em, baseline: 15%)
    #set text(size: 20pt)
    Let $bold(Z)$ be a configuration of the system over $T$, then
        we want to ensure for $A := {wind, sol, coal, nuc}$:\
    #align(center)[
        $forall t :
        sum_(a in A) p_(t, a)(bold(Z))- bold(tau)_t approx 0$]
        
        $forall t in [T - 1], a in A :
        bold(Z)_(t, a) - bold(Z)_(t+1,a) <= 1$
    While minimizing:
    #align(center)[
        $C(bold(Z)) =
        min_bold(Z) sum_(t=1)^T sum_(a in A) c_(t, a)(bold(Z))$
        
        $W(bold(Z))=
        min_bold(Z) sum_(t=1)^(T-1) sum_(a in A)
        s_(a)(bold(Z)_(t,a),bold(Z)_(t+1,a))$
        
        $O(bold(Z))=
        min_bold(Z) sum_(t=1)^T sum_(l in"Lines")
        bold(1){l" not overloaded"}$
    ]
    #uncover("2-")[
        #place(left + horizon, dx: -3em)[
            #block(fill: white, stroke: 5pt, inset: 1em)[
                #text(weight: "bold", "Optimisation Problem")]]
    ]
]
