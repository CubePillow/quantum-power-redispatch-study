#import "/globals.typ": *

#cetz.canvas({
        import cetz.draw: *
            let ymax = 3
            let xmax = 5.5
        let icon-h = 30pt

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
            (1.5, ymax / 2), image("../img/city.svg", height: icon-h),
            name: "city1"
        )
        content(
            (xmax - 1.5, ymax / 2),
            image("../img/city.svg", height: icon-h),
            name: "city2"
        )

        line("sol.south", "city1.north-west", stroke: 4pt)
        line("wind.north", "city1.south-west", stroke: 4pt)
        line("nuc.south", "city2.north-east", stroke: 4pt)
        line("coal.north", "city2.south-east", stroke: 4pt)

            line("city1.east", "city2.west", stroke: 4pt, name: "line")
        })
