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
#slide(composer: (2.25fr, 3fr), gutter: 2em, repeat: 2)[
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
    #pause
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
== Modifying $bold(x)$ with Cycles
#slide(composer: (2.25fr, 3fr), gutter: 2em, repeat: 5)[
    #side(1)
][
    #align(left)[
        #set text(size: 17pt)
        #only("1-3")[
    #cetz.canvas({
        import cetz.draw: *
        content(
            (0, 0),
            [
                $
                    #curx = &mat(
                1, 0, 0, 1, 0, 0, dots.h.c, 1, 0, 0, dots.h.c
                )\
                c_1 compose c_2 compose #curx = &mat(
                0, 1, 0, 0, 0, 1, dots.h.c, 1, 0, 0, dots.h.c
                ) $
            ],
            name: "x"
        )
        // arc-through(
        //     (3.25, 0.5), (3.65, 0), (3.25, -0.5),
        //     style: (mark: (end: "<>"))
        // )
        arc-through((-1.5, -0.85), (-1.25, -1), (-1, -0.85))
        content((rel: (-0.2, -0.75)), [$C_1$])
        arc-through((0.25, -0.85), (0.85, -1), (1.45, -0.85))
        content((rel: (-0.45, -0.75)), [$C_2$])
    })]
        #only(4)[
    #cetz.canvas({
        import cetz.draw: *
        content(
            (0, 0),
            [
                $
                    #curx = &mat(
                1, 0, 0, 1, 0, 0, dots.h.c, 1, 0, 0, dots.h.c
                )\
                c_1 compose c_2 compose #curx = &mat(
                0, 0, 1, 0, 0, 1, dots.h.c, 1, 0, 0, dots.h.c
                ) $
            ],
            name: "x"
        )
        // arc-through(
        //     (3.25, 0.5), (3.65, 0), (3.25, -0.5),
        //     style: (mark: (end: "<>"))
        // )
        arc-through((-1.5, -0.85), (-1.25, -1), (-0.25, -0.85))
        content((rel: (-0.44, -0.75)), [$C_1$])
        arc-through((2.75, -0.85), (3.85, -1), (4.25, -0.85))
        content((rel: (-0.44,-0.75)), [Not adjacent state])
    })]
    ]
    #set text(size: 17pt)
    #only(1)[
        What is a cycle? 
        $
            c_1 vec(x,y,z) = mat(0,1,0; 1,0,0; 0,0,1) vec(x,y,z)
            =vec(y,x,z)
        $
        $
            C_1 #curx =
            mat(c_1, bold(O); bold(O), bold(I)) vec(1, 0, 0, dots.v)
            = vec(0, 1, 0, dots.v)
        $
    ]
    #only(2)[
        Disjoint cycles are commutative
        $
            &C_1 C_2 = C_1 C_2 
            = mat(c_1, bold(O); bold(O), bold(I))
            mat(bold(I_3), bold(O), bold(O);
            bold(O), c_2, bold(O);
            bold(O), bold(O), bold(I))
                =mat(c_1, bold(O), bold(O);
            bold(O), c_2, bold(O);
            bold(O), bold(O), bold(I))
        $
        where
        $
            c_1 = mat(0,1,0;1,0,0;0,0,1) #h(2em)
                c_2 = mat(0,0,1;0,1,0;1,0,0)
        $
    ]
    #only(3)[
        Non-Disjoint cycles not commutative
        $
            &mat(0,1,0;1,0,0;0,0,1)  mat(0,0,1;0,1,0;1,0,0) vec(x,y,z)
            = vec(y, z, x)\
            !=&mat(0,0,1;0,1,0;1,0,0)mat(0,1,0;1,0,0;0,0,1)
            vec(x,y,z) = vec(z,x,y)
        $
        Prevents creation of the subproblem in step 2 
    ]
    #only(4)[
        Rectifying cycles to ensure adjacent state switches
        #set align(center)
        #cetz.canvas({
            import cetz.draw: *
            content(
                (0, 0),
                [$
                    mat(1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0)\
                    arrow.b.stroked\
                    mat(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0)
                $],
                name: "x"
            )
            for i in range(5) {
                line((-4.3 + (i * 1.8), 1.4), (-2.9 + (i * 1.8), 1.4))
                content((rel: (-0.65, 0.4)), [$t=#(i+1)$])
                content((rel: (0, 0.6)), [#if (i==2) {$3$} else {$1$}])
            }
            for i in range(5) {
                line((-4.3 + (i * 1.8), -1.3), (-2.9 + (i * 1.8), -1.3))
                content((rel: (-0.65, -0.4)), [$#(3 - calc.abs(2 - i))$])
            }
        })
        1 + 2 rectified cycles forms a permutation matrix.
    ]
    #only(5)[
        #set text(size: 25pt)
        #set align(center + horizon)
        Generate random rectified permutation matrices to ensure adjacent states swaps between timesteps; while ensuring all permutation matrices are disjoint.
    ]
]
== Subproblem of Selecting Cycles
#slide(composer: (2fr, 3fr))[
    #side(2)
][
    #set text(size: 20pt)
    We have a set of rectified permutation matrices
    $
        CC = {C_1, dots, C_c}
    $
    Now need to create an optimisation problem that selects which
    matrices to apply
    $
        op("arg min")_{bold(alpha) in {0, 1}} bold(alpha)^top B bold(alpha)\
        [B]_(i,j) = cases(
        E(c_i^*, c_j^*)  &"if" i!=j,
        E(c_i^*, c_j^*) + E(c_i^*, bold(x)) + E(bold(x), c_j^*)
            &"otherwise"
        )
    $
    where: $#h(2em) E(bold(v), bold(u)) = bold(v)^top Q bold(u)
    #h(3em) c_i^* = (C_i - bold(I)) bold(x)$
]
#slide(composer: (2.25fr, 3fr), gutter: 2em)[
    #side(3)
][
    #set text(size: 20pt)
    #set align(horizon)
    After solving for $bold(alpha)$ with $CC = {C_1,dots,C_c}$
    $
        op("arg min")_{bold(alpha) in {0, 1}} bold(alpha)^top B bold(alpha)
    $
    We apply the selected permutations to our current solution
    #curx before the next iteration
    $
        #curx = (product_(i=1)^c alpha_i C_i) #curx
    $
]
== $alpha$-expansion
#slide(composer: (2.25fr, 3fr), gutter: 2em)[
    #side(4)
][
    #set text(size: 20pt)
    #set align(horizon)
    With $alpha$-expansion we avoid having to consider constraints
    on the solution
      $
          op("arg min", limits: #true)_bold(x)
              &bold(x)^top
              (mu_C Q_C + mu_S Q_S + mu_O Q_O + lambda_p Q_P) bold(x)
      $
      #table(columns: (1fr, 3fr), stroke: none, align: (right, left),
          $Q_C:=$, [Production Cost],
          $Q_S:=$, [Switching Cost],
          $Q_O:=$, [Unbalanced Penalty for number of overloaded lines],
          $Q_P:=$, [Constraint on production targets],
      )
]
