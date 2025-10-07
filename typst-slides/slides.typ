#import "/globals.typ": *

#show: iais-theme.with(
    aspect-ratio: "16-9",
    logo: logo_iais,
    footer: logo_iais,
    config-info(
        title: [Multi-Objective Quantum Power System Redispatch],
        authors: (
            "Loong Kuan Lee",
            "Thore Gerlach",
            "Johannes Knaute",
            "Florian Gerhardt",
            "Patrick Völker",
            "Tomislav Maras",
            "Alexander Dotterweich",
            "Nico Piatkowski"),
        institutions: (logo_pwc_big, logo_iais_big),
    ),
)


#set heading(numbering: numbly("{1}.", default: "1.1"))
#title-slide()

#include "typ/intro.typ"

#focus-slide[
    #set text(weight: "semibold")
    Goal:\
    Investigate the possibility of\
    using Quantum Computing for the\
    Power System Redispatch Problem
]

#include "typ/qa.typ"

#include "typ/qubo.typ"

#focus-slide[
    #set text(weight: "semibold", size: 30pt)
    In real world scenarios we can expect around:
    #align(center)[
        $~338$ controllable resources\
        $~5$ states each\
        $=> |bold(x)| = 1690 * T$
    ]
    For large $T$ can be challenging for even a classical computer,
    let alone current day small and noisy quantum computers.
]

#include "typ/decomp.typ"
#include "typ/moo.typ"
#include "typ/exp.typ"
