// This theme is modified from https://github.com/andreasKroepelin/polylux/blob/main/themes/simple.typ
// Author of Simple Theme: Andreas Kröpelin
// Author of Modifications: Loong Kuan Lee

#import "@preview/touying:0.6.1": *

#let current-progressive-outline(
  self: none,
  alpha: 60%,
  level: auto,
  numbered: (false,),
  filled: (false,),
  paged: (false,),
  numbering: (),
  text-fill: none,
  text-size: none,
  text-weight: none,
  vspace: none,
  title: none,
  indent: (0em,),
  fill: (repeat[.],),
  short-heading: true,
    section-num: 1,
    num-sub: 1,
  uncover-fn: body => body,
  ..args,
) = components.progressive-outline(
  alpha: alpha,
  level: level,
  transform: (cover: false, alpha: alpha, ..args, it) => {
    let array-at(arr, idx) = arr.at(idx, default: arr.last())
    let set-text(level, body) = {
      set text(fill: {
        let text-color = if type(text-fill) == array and text-fill.len() > 0 {
          array-at(text-fill, level - 1)
        } else {
          text.fill
        }
          if cover {
              self.colors.neutral-light
          // utils.update-alpha(text-color, alpha)
        } else {
          text-color
        }
      })
      set text(
        size: array-at(text-size, level - 1),
      ) if type(text-size) == array and text-size.len() > 0
      set text(
        weight: array-at(text-weight, level - 1),
      ) if type(text-weight) == array and text-weight.len() > 0
      body
    }
      let body = {
      if type(vspace) == array and vspace.len() > it.level - 1 {
        v(vspace.at(it.level - 1))
      }
      h(range(1, it.level + 1).map(level => array-at(indent, level - 1)).sum())
      set-text(
        it.level,
        {
          if array-at(numbered, it.level - 1) {
            let current-numbering = numbering.at(it.level - 1, default: it.element.numbering)
            if current-numbering != none {
              _typst-builtin-numbering(
                current-numbering,
                ..counter(heading).at(it.element.location()),
              )
              h(.3em)
            }
          }
          link(
            it.element.location(),
            {
              if short-heading {
                utils.short-heading(self: self, it.element)
              } else {
                it.element.body
              }
                let cur-sub = counter(heading).at(it.element.location()).last()
                if (cur-sub < num-sub - 1) {
              text(fill: rgb("#000000"), weight: "black", "  ·  ")
                }
              // text(fill: rgb("#000000"), weight: "black", "  ·  ")
              // box(
              //   width: 1fr,
              //   inset: (x: .2em),
              //   if array-at(filled, it.level - 1) {
              //     array-at(fill, level - 1)
              //   },
              // )
              if array-at(paged, it.level - 1) {
                _typst-builtin-numbering(
                  if page.numbering != none {
                    page.numbering
                  } else {
                    "1"
                  },
                  ..counter(page).at(it.element.location()),
                )
              }
            },
          )
        },
      )
      }
      if (counter(heading).at(it.element.location()).first() != section-num) {
          ""
      }
    else if cover {
      body
    } else {
      uncover-fn(body)
    }
  },
  title: title,
  ..args,
)


/// Default slide function for the presentation.
///
/// - config (dictionary): The configuration of the slide. You can use
/// `config-xxx` to set the configuration of the slide. For more several
/// configurations, you can use `utils.merge-dicts` to merge them.
///
/// - repeat (int, auto): The number of subslides. Default is `auto`,
/// which means touying will automatically calculate the number of
/// subslides.
///
////   The `repeat` argument is necessary when you use `#slide(repeat:
////   3, self => [ .. ])` style code to create a slide. The
////   callback-style `uncover` and `only` cannot be detected by touying
////   automatically.
///
/// - setting (function): The setting of the slide. You can use it to
/// add some set/show rules for the slide.
///
/// - composer (function): The composer of the slide. You can use it to
/// set the layout of the slide.
///
///   For example, `#slide(composer: (1fr, 2fr, 1fr))[A][B][C]` to split
///   the slide into three parts. The first and the last parts will take
///   1/4 of the slide, and the second part will take 1/2 of the slide.
///
///   If you pass a non-function value like `(1fr, 2fr, 1fr)`, it will
///   be assumed to be the first argument of the
///   `components.side-by-side` function.
///
///   The `components.side-by-side` function is a simple wrapper of the
///   `grid` function. It means you can use the `grid.cell(colspan: 2,
///   ..)` to make the cell take 2 columns.
///
///   For example, `#slide(composer: 2)[A][B][#grid.cell(colspan:
///   2)[Footer]]` will make the `Footer` cell take 2 columns.
///
///   If you want to customize the composer, you can pass a function to
///   the `composer` argument. The function should receive the contents
///   of the slide and return the content of the slide, like
///   `#slide(composer: grid.with(columns: 2))[A][B]`.
///
/// - bodies (array): The contents of the slide. You can call the
/// `slide` function with syntax like `#slide[A][B][C]` to create a
/// slide.

#let slide(title: auto, composer: auto, gutter: auto, ..args) = touying-slide-wrapper(self => {
    if title != auto {
        self.store.title = title
    }
    // set page
    let header(self) = context {
        let h = utils.current-heading()
        let h-count = counter(heading).at(h.location())
        let num-sub = h-count.last()
        let rest = query(heading.where().after(heading.where(body: h.at("body"))))
        for htmp in rest {
            if htmp.at("depth") < 2 {
                break
            }
            num-sub += 1
        }
        let sec-num = h-count.first()
        set align(top)
        show: pad.with(x: 0em, bottom: 0em, top:39pt)
        show: components.cell.with(inset: 35pt)
        set align(horizon)
        set par(spacing: 17pt)
        set text(size: 27pt, weight: "bold", font: "Open Sans")
        utils.display-current-heading(
            level: 1, style: h => str(sec-num) + ". " + h.body)
        linebreak()
        set text(
            size: 17pt, fill: self.colors.primary, weight: "semibold"
        )
        if self.store.title != none {
            utils.call-or-display(self, self.store.title)
        } else {
            current-progressive-outline(
                self: self, level: 2, target: heading.where(level: 2),
                section-num: sec-num, num-sub: num-sub
            )

        }
    }
    let footer(self) = {
        set align(bottom)
        set align(center)
        show: components.cell.with(
            // fill: self.colors.primary,
            width: 92.5%,
            stroke: (top: (paint: self.colors.primary, thickness: 2.5pt)),
            // inset: (bottom: 0.75em, top:0em)
        )
        set align(left)
        set text(
            fill: self.colors.neutral-light, size: 8pt,
            weight: "semibold"
        )
        let tmp = context "Slide " + utils.slide-counter.display() + "\n" + sym.copyright + " Fraunhofer Institute for Intelligent Analysis and Information Systems IAIS"
        grid(
            columns: (100%, 1fr, 1fr),
            align: (left, right),
            inset: ((bottom: 17pt), (bottom: 7pt)),
            tmp,
            utils.call-or-display(self, self.store.footer),
        )
    }
    self = utils.merge-dicts(
        self,
        config-page(
            header: header,
            footer: footer,
        ),
    )
    touying-slide(
        self: self,
        composer: components.side-by-side.with(
            columns: composer, gutter: gutter
        ),
        ..args)
})

/// Centered slide for the presentation.
/// 
/// - config (dictionary): The configuration of the slide. You can use `config-xxx` to set the configuration of the slide. For more several configurations, you can use `utils.merge-dicts` to merge them.
#let centered-slide(config: (:), ..args) = touying-slide-wrapper(self => {
    touying-slide(self: self, ..args.named(), config: config, align(center + horizon, args.pos().sum(default: none)))
})



#let title-slide(..args) = touying-slide-wrapper(self => {
  let info = self.info + args.named()
    let body = {
      set align(left + top)
      text(
          size: 30pt, fill: self.colors.neutral-darkest,
          weight: "bold", info.title)
      linebreak()
    set text(fill: self.colors.neutral-darkest, weight: "bold")
      if info.authors != none {
          text(info.authors.join(", "))
      }
      // set align(right + bottom)
      place(right + bottom, dy: 25pt)[
      #grid(columns: info.institutions.len(),
          gutter: 40pt,
          ..info.institutions)
      ]
      // grid(columns: info.institutions.len(),)
  }
  touying-slide(self: self, body)
})

/// New section slide for the presentation. You can update it by updating the `new-section-slide-fn` argument for `config-common` function.
/// 
/// - config (dictionary): The configuration of the slide. You can use `config-xxx` to set the configuration of the slide. For more several configurations, you can use `utils.merge-dicts` to merge them.
#let new-section-slide(config: (:), body) = centered-slide(config: config, [
    #set heading(offset: 2)
    #text(30pt, weight: "bold", utils.display-current-heading(level: 1))

    #body
])


/// Focus on some content.
///
/// Example: `#focus-slide[Wake up!]`
/// 
/// - config (dictionary): The configuration of the slide. You can use `config-xxx` to set the configuration of the slide. For more several configurations, you can use `utils.merge-dicts` to merge them.
/// 
/// - background (color, auto): The background color of the slide. Default is `auto`, which means the primary color of the slides.
/// 
/// - foreground (color): The foreground color of the slide. Default is `white`.
#let focus-slide(config: (:), background: auto, foreground: black, body) = touying-slide-wrapper(self => {
    self = utils.merge-dicts(
        self,
        config-common(freeze-slide-counter: true),
        config-page(fill: if background == auto {
            self.colors.neutral-lightest
        } else {
            background
        }),
    )
    set text(fill: foreground, size: 1.75em)
    touying-slide(self: self, config: config, align(center + horizon, body))
})


// Theme
#let iais-theme(
    aspect-ratio: "16-9",
    footer: none,
    logo: none,
    ..args,
    body,
) = {
    set text(size: 25pt, font: "Open Sans")
    show: touying-slides.with(
        config-page(
            paper: "presentation-" + aspect-ratio,
            margin: (top: 100pt, bottom: 65pt, x:40pt),
            header-ascent: 40pt,
            // footer-descent: 0pt,
        ),
        config-common(
            slide-fn: slide,
            new-section-slide-fn: new-section-slide,
        ),
        config-colors(
            neutral-light: rgb("#a8afaf"),
            neutral-lightest: rgb("#ffffff"),
            neutral-darkest: rgb("#000000"),
            primary: rgb("#1f82c0"),
        ),
        config-methods(alert: utils.alert-with-primary-color),
        // save the variables for later use
        config-store(title: none, footer: footer, logo: logo),
        ..args,
    )

    body
}
