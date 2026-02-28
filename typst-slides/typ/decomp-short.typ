#import "/globals.typ": *

#let curx = [$bold(hat(x))$]

#let hl(cond, body) = {[
    #set text(fill: if (cond) { black } else {gray})
    #body
]}

#let side(pos) = {[
    #v(0.5em)
    $alpha$-expansion

    #set enum(indent: 1em, spacing: 1em)
    #set text(size: 17pt)
    #hl((pos == 1) or (pos == 4))[
        + Generate random modifications to current solution vector #curx
          $ CC := {C_1, dots, C_n} $
    ]
    #hl((pos == 2) or (pos == 4))[
        + Find subset of $CC$ that
          minimises the objective function
          $
              op("arg min")_(hat(C) subset.eq CC) f(hat(C) compose #curx)
          $
    ]
    #hl((pos == 3) or (pos == 4))[
        + Apply selected modifications
    ]
    #hl(pos == 4)[
        + Repeat until convergence
    ]
]}

#let cycles-ex() = {
    align(center)[
    #cetz.canvas({
        import cetz.draw: *
        content(
            (0, 0),
            [
                $mat(
                0, 1, 0, 0, 0, 1, dots.h.c, 0, 0, 1, dots.h.c
                )$
            ],
            name: "x"
        )
        arc-through((-2.75, -0.25), (-2.5, -0.4), (-2.25, -0.25))
        content((rel: (-0.2, -0.5)), [$c_1$])
        arc-through((-1.15, -0.25), (-0.65, -0.4), (-0.15, -0.25))
        content((rel: (-0.45, -0.5)), [$c_2$])
        arc-through((1.1, -0.25), (1.65, -0.4), (2.15, -0.25))
        content((rel: (-0.45, -0.5)), [$c_n$])
    })]
}

= Problem Decomposition with $alpha$-expansion
== Core Idea
#slide(composer: (2.25fr, 3fr), gutter: 2em, repeat: 1)[
    #align(center + horizon)[
        #v(0.5em)    
    #include "ex-prob.typ"
    #let s = 0.9em
    #text(size: 17pt)[
        Assume we have some initial solution #curx that we start with:]
    #cetz.canvas({
        import cetz.draw: *
        content(
            (0, 0),
            [
                $#curx = mat(
                1, 0, 0, 1, 0, 0, dots.h.c, 1, 0, 0, dots.h.c
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
    ]
][
    #set text(size: 17pt)
    #set align(horizon)
    1. Generate random modifications to #curx, $CC$
       #align(center)[
    #cetz.canvas({
        import cetz.draw: *
        content(
            (0, 0),
            [
                $mat(
                0, 1, 0, 0, 0, 1, dots.h.c, 0, 0, 1, dots.h.c
                )$
            ],
            name: "x"
        )
        arc-through((-2.75, -0.25), (-2.5, -0.4), (-2.25, -0.25))
        content((rel: (-0.2, -0.5)), [$c_1$])
        arc-through((-1.15, -0.25), (-0.65, -0.4), (-0.15, -0.25))
        content((rel: (-0.45, -0.5)), [$c_2$])
        arc-through((1.1, -0.25), (1.65, -0.4), (2.15, -0.25))
        content((rel: (-0.45, -0.5)), [$c_n$])
    })]
    2. Find subset of $CC := {c_1, dots, c_n}$ that minimises\
       the objective function:
       $
           op("arg min")_(hat(C) subset.eq CC)
           (hat(C) compose #curx)^top Q (hat(C) compose #curx)
       $
    
    3. Apply selected modifications:
       $
           #curx = hat(C) compose #curx
       $
    4. Repeat until convergence
]
