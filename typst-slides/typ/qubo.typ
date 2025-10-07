#import "/globals.typ": *

#let prob() = {
align(center + horizon)[
    #include "ex-prob.typ"
    #text(size: 17pt)[
        Binary encoding of a solution for\
        our example assuming\
        each resource has $2$ states with $T = 2$:]
    #cetz.canvas({
        import cetz.draw: *
        content(
            (0, 0),
            [
                $bold(x) = mat(1, 0, 0, 1, 1, 0, 1, 0, dots, 0, 1)$
            ],
            name: "x"
        )
        for (i, t) in (wind, nuc, sol, coal).enumerate() {
            line((rel: (2.2 + (1.57 * i), -0.5), to: "x.west"),
                (rel: (3.2 + (1.57 * i), -0.5), to: "x.west"))
            content((rel: (-0.5, -0.7)), t(width: 25pt))
            content((rel: (-0.05, -0.8)), text(size: 12pt, [$t=1$]))
        }
        line((rel: (-0.4, -0.5), to: "x.east"),
            (rel: (-1.4, -0.5), to: "x.east"))
        content((rel: (0.5, -0.7)), coal(width: 25pt))
        content((rel: (-0.05, -0.8)), text(size: 12pt, [$t=2$]))
    })
]}

= Formulating the Redispatch Problem as a QUBO
== Encoding System Configuration
#slide(composer: (2.25fr, 3fr), gutter: 2em, repeat: 1)[
    #prob()
][
    #set align(center + horizon)
    #set text(size: 20pt)
        Standard Technique:
        Incoporate constraints\ with 
        Lagrangian multipliers
        $
            "minimise"& bold(x)^top Q bold(x)\
            "subject to"& bold(x)^top Q_C bold(x) - m = 0
        $
        $ 
            arrow.b.stroked\
            "minimize" bold(x)^top (Q + lambda_C Q_C )  bold(x)
        $
        Goal: Express all objective or contraints as a linear or quadratic function of $bold(x)$.
]


#let prob3() = {
    align(center + horizon)[
        #v(0.5em)    
    #include "ex-prob.typ"
    #let s = 0.9em
    #text(size: 17pt)[
        Assume $T=2$ and
        each resource in\
        $R:={
        #wind(height: s), #nuc(height: s),
        #sol(height: s), #coal(height:s)}$
        has $3$ states:]
    #cetz.canvas({
        import cetz.draw: *
        content(
            (0, 0),
            [
                $bold(x) = mat(
                1, 0, 0, 0, 1, 0, dots.h.c, 0, 1, 0, dots.h.c
                )$
            ],
            name: "x"
        )
        for (i, t) in (wind, nuc).enumerate() {
            line((rel: (2.2 + (2.35 * i), -0.5), to: "x.west"),
                (rel: (4 + (2.35 * i), -0.5), to: "x.west"))
            content((rel: (-0.9, -0.7)), t(width: 25pt))
            content((rel: (0, -0.8)), text(size: 12pt, [$t=1$]))
        }
        line((rel: (-3.15, -0.5), to: "x.east"),
            (rel: (-1.2, -0.5), to: "x.east"))
        content((rel: (-0.9, -0.7)), wind(width: 25pt))
        content((rel: (0, -0.8)), text(size: 12pt, [$t=2$]))
    })
]}


== Constraints on $bold(x)$
#slide(composer: (2.25fr, 3fr), gutter: 2em, repeat: 2, self => [
    #only("1")[
        #prob()
    ]
    #only(2)[
        #prob3()
    ]
],self=>[
    #let (uncover, only, alternatives) = utils.methods(self)
    #set text(20pt)
    #set align(center)
    #only(1)[
    One-hot constraint: 
    $
        op("argmin")_bold(x) bold(x)^top Q_H bold(x)\
        H := mat(-1, 1; 1, -1)#h(2em)
        Q_H := mat(
        H, dots.h.c, 0;
        dots.v, dots.down, dots.v;
        0, dots.h.c, H
        )
    $
        $
            #h(2.5em)
            vec(1, 1)^top mat(-1, 1; 1, -1) vec(1,1) &= 0\
            vec(1, 0)^top mat(-1, 1; 1, -1) vec(1,0) &= -1
        $
    ]
    #only(2)[
    State Switching constraint: 
    $
        op("argmin")_bold(x) bold(x)^top Q_A bold(x)\
        A := mat(0, 0, 1; 0, 0, 0; 1, 0, 0)#h(2em)
        // Q_H := mat(
        // 0, #only("4-")[#text(fill: red, "A")]#only("3")[A], dots.h.c, 0;
        // dots.v,dots.v, dots.down, dots.v;
        // 0, 0, dots.h.c, A;
        // 0, 0, dots.h.c, 0
        // )
        Q_H := mat(
        bold(O)_(4 times 3) ,
        mat(delim: #none,
        #only("4-")[#text(fill: red, [$A$])]#only("3")[$A$],
        dots.h.c, 0; dots.v, dots.down, dots.v; 0, dots.h.c, A
        );
        bold(O)_(4 times 3), bold(O)_(4 times 3))
    $
    $
        #h(2.5em)
            &vec(1, 0, 0)^top
            mat(0, 0, 1; 0, 0, 0; 1, 0, 0) &&vec(0, 0, 1)
            = 1\
            #box(wind(height: 0.8em), baseline: 10%)
            &t=1 &#box(wind(height: 0.8em), baseline: 10%) &t=2
    $
    ]
])
== Power Target $tau$
#slide(composer: (2.25fr, 3fr), gutter: 2em, repeat: 1)[
    #prob3()
][
    #set align(horizon)
    #set text(size: 18pt)
    #let s = 0.9em
    We want the following:
    $
        vec(
        sum_(a in R)sum_(i=1)^3 p_(1,a,i)bold(x)_(1,a,i),
        sum_(a in R)sum_(i=1)^3 p_(2,a,i)bold(x)_(1,a,i)
        ) approx
        vec(tau_1, tau_2)
    $
    So we need to minimise
    $
        P(bold(x)) =
        sum_(t=1)^T (sum_(a in R)sum_(i=1)^3
        p_(t,a,i)bold(x)_(1,a,i) - tau_t)^2
    $
    which is a quadratic of the variables in $bold(x)$.\
    Therefore can be directly expressed as a QUBO objective.
]
== Objectives to Minimise
#let inlineFrac(a, b) = {
    return $#super(a) #h(-1pt) slash #h(-1pt) #sub(b)$ }
#slide(composer: (2.25fr, 3fr), gutter: 2em, repeat: 8)[
    #only("1,2,8")[
    #prob3()
    ]
    #set align(center)
    #only("3-4")[
        #set text(17pt)
        #v(0.735em)    
        #include "ex-prob.typ"        
    ]
    #set text(size: 15pt)
    #let xs = range(-20, 20).map(x => x/10)
    #only("5")[
        #let ys1 = xs.map(x => 1 / (1 + calc.exp(10*x)))
        #let ys2 = xs.map(x => if (x <= 0) {1} else {0})
        #show lq.selector(lq.legend): set grid(gutter: 5pt)
        #lq.diagram(
            lq.plot(xs, ys1, mark: none, smooth: true,
                label: [$1 / (1 + e^(10x))$]
            ),
            lq.plot(xs, ys2, mark: none, smooth: true,
                label: [$[x <= 0]$]
            ),
            // lq.plot(xs, ys3, mark: none, smooth: true,
            //     label: [$[x < 0]$]
            // ),
            width: 8cm, height: 3.03cm,
            yaxis: (position: -2.1, tip: tiptoe.stealth),
            xaxis: (position: -0.1, tip: tiptoe.stealth, label: [$h(x)$]),
            grid: none,
        )
    ]
    #only("6-7")[
        #set text(size: 15pt)
        #let xs = range(-20, 20).map(x => x/10)
        #let ys1 = xs.map(x => calc.exp(-1 * x))
        #let ys2 = xs.map(x => 1 - x + calc.pow(x, 2) / 2)
        #let ys3 = xs.map(x => if(x < 0) { 1 } else {0})
        #lq.diagram(
            lq.plot(xs, ys1, mark: none, smooth: true,
                label: [$e^(-h(x))$]
            ),
            lq.plot(xs, ys2, mark: none, smooth: true,
                label: [$1 - h(x) + 1/2 h(x)^2$]
            ),
            // lq.plot(xs, ys3, mark: none, smooth: true,
            //     label: [$[x < 0]$]
            // ),
            width: 8cm, height: 3.215cm,
            yaxis: (position: -2.1, tip: tiptoe.stealth),
            xaxis: (position: -0.1, tip: tiptoe.stealth, label: [$h(x)$]),
            grid: none,
        )
    ]


    #only("3-7")[
        // #set align(center)
        #set text(17pt)
        // #v(0.735em)    
        // #include "ex-prob.typ"
        #v(0.5em)
        #table(
            columns: (1fr, 2fr),
            align: (right, left),
            stroke: none,
            row-gutter: 0.75em,
            column-gutter: 0em,
            $P(bold(x))_(t,a):=$,
            $sum_(i=1)^k p_(t,a,i)bold(x)_(t,a,i)$,
            $[M]_(t,l):=$,
            [Maximum Capacity of line $l$ at time $t$],
            $[S]_(a,l):=$,
            [Effect of resource $a$ on transmission line $l$],
        )
        // #grid(
        //     columns: 2, gutter: 1em,
        //     $S:=$,    
        // )
        // #grid(
        //     columns: 2, gutter: 1em,
        //     $P(bold(x))_(t,a)$,
        //     $sum_(i=1)^k p_(t,a,i)bold(x)_(t,a,i)$
        // )
        // $P(bold(x))_(t,a) := sum_(i=1)^k p_(t,a,i)bold(x)_(t,a,i)$
    ]
][
    #set text(size: 18pt)
    #only(1)[
        #align(center)[
            Costs of the solution $bold(x)$ are straightforward
        ]

        Production Cost:
        $
            C(bold(x)) = sum_(t=1)^T (sum_(a in R)sum_(i=1)^3
            c_(t,a,i) bold(x)_(t,a,i))
            = bold(x)^top op("diag")(bold(c)) bold(x)
        $
        Switching Cost:
        $
            W(bold(x)) = sum_(t=1)^(T-1) sum_(a in R)sum_(i=1)^3sum_(j=1)^3
            bold(x)_(t,a,i) bold(x)_(t+1,a,j) w_(a)(i, j)
        $
        where $w_(a)(i,j)$ is the cost of resource $a$ switching from
        $i$ to $j$.
    ]
    #only("2-7")[
        Number of overloaded lines more complicated
    ]
    #only("3-7")[
        $
            O(bold(x)) = sum_(t=1)^T sum_l^L
            cases(
            1  &[P(bold(x))S]_(t,l) >= [M]_(t,l),
            0  &"otherwise")
        $
    ]
    #only("4-6")[
        So we need to find a way to encode\
        $[[P(bold(x))S]_(t,l) >= [M]_(t,l)]$ as a quadratic function of
        $bold(x)$.
    ]
    
    #only("5-6")[
        Let $h(bold(x))_(t,l) = [M]_(t,l) - [P(bold(x))S]_(t,l)$, then
    ]
    #only("5")[
        $
            [h(bold(x))_(t,l) <= 0]
            approx 1 slash (1 +e^(c dot h(bold(x))_(t,l)))
        $
    ]
    #only("6")[
        $
            &[h(bold(x))_(t,l) <= 0]
            approx 1 slash (1 +e^(c dot h(bold(x))_(t,l)))\
                &approx e^(-c dot h(bold(x))_(t,l))\
                &approx 1 - c dot h(bold(x))_(t,l)
                + 1 / 2 [c dot h(bold(x))_(t,l)]^2
        $
    ]
    #only("7")[
        Let $h(bold(x))_(t,l) = [M]_(t,l) - [P(bold(x))S]_(t,l)$ then by
        using unbalanced penalisation we get
        $
            O(bold(x)) = sum_(t=1)^T sum_(l=1)^L
            1 - c dot h(bold(x))_(t,l) + 1 / 2 [c dot h(bold(x))_(t,l)]^2
        $
        which is a quadratic function of $bold(x)$ since
        $[P(bold(x))S]_(t,l)$ is a linear function of $bold(x)$.
    ]
    #only(8)[
        Final QUBO problem:
        $
            op("arg min")_bold(x)
                &bold(x)^top (mu_C Q_C + mu_S Q_S + mu_O Q_O) bold(x)\
                &+lambda_P bold(x)^top Q_P bold(x)
                +bold(x)^top (lambda_H Q_H + lambda_A Q_A) bold(x)
        $
        where the QUBO objectives are labeled as:
        #table(columns: (1fr, 3fr), stroke: none, align: (right, left),
            $Q_C:=$, [Production Cost],
            $Q_S:=$, [Switching Cost],
            $Q_O:=$, [Unbalanced Penalty for number of overloaded lines],
            $Q_P:=$, [Constraint on production targets],
            $Q_H:=$, [One-Hot Constraint for $bold(x)$],
            $Q_A:=$, [Adjacent state swap constraint],
        )
    ]
]
