#import "@preview/touying:0.6.1": *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/muchpdf:0.1.1": muchpdf
#import "@preview/cetz:0.4.2"
#import "@preview/tiptoe:0.3.1"
#import "@preview/lilaq:0.5.0" as lq
#import "@preview/suiji:0.4.0": *
#import "iais.typ": *

// #let logo_iais = image("IAIS_logo.png", fit: "contain", height: 0.5em)
// #let logo = image("IAIS_logo.png", fit: "contain", height: 10em)
#let logo_iais = image("IAIS_logo.png", width: 4cm)
#let logo_pwc = image("pwc-logo.png", width: 2.5cm)

#let logo_iais_big = image("IAIS_logo.png", height: 2.5em)
#let logo_pwc_big = image("pwc-logo.png", height: 2.5em)

#let sol(..args) = {
    image("img/solar.svg", ..args)
}
#let wind(..args) = {
    image("img/wind.svg", ..args)
}
#let nuc(..args) = {
    image("img/nuclear.svg", ..args)
}
#let coal(..args) = {
    image("img/coal.svg", ..args)
}
