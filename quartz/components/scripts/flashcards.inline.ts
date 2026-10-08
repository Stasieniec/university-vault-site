// Study mode for flashcards written as collapsible callouts:
//
//   > [!card]- Question
//   > Answer
//
//   > [!exam]- Exam-style question
//   > The points the answer has to hit
//
// The callouts already work as click-to-reveal without this script. This adds a launcher
// above each set and a drill view: one card at a time, "missed" cards come back a few cards
// later until each has been answered correctly once. State is kept in sessionStorage, so it
// survives reloads and navigation inside the tab and is gone when the tab closes.

type Card = { id: string; q: string; a: string; exam: boolean; deck: string }
type Deck = { label: string; cards: Card[]; anchor: Element }
type State = { ids: string[]; queue: string[]; done: string[]; misses: Record<string, number> }

const CARD_SELECTOR = '.callout[data-callout="card"], .callout[data-callout="exam"]'

function hash(str: string): string {
  let h = 5381
  for (let i = 0; i < str.length; i++) h = ((h << 5) + h + str.charCodeAt(i)) | 0
  return (h >>> 0).toString(36)
}

function shuffle<T>(xs: T[]): T[] {
  const a = xs.slice()
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

function load(key: string): State | null {
  try {
    const raw = sessionStorage.getItem(key)
    return raw ? (JSON.parse(raw) as State) : null
  } catch {
    return null
  }
}

function save(key: string, state: State) {
  try {
    sessionStorage.setItem(key, JSON.stringify(state))
  } catch {
    // Private mode or storage disabled: the session still works, it just won't survive a reload.
  }
}

function forget(key: string) {
  try {
    sessionStorage.removeItem(key)
  } catch {}
}

// On a course page the cards arrive through transclusions, one per lecture, each under its own
// heading. On a lecture page there is no transclusion: the note's short cards are one deck and
// its long-form exam questions another, so a spare-moment session never hits a model answer.
function deckLabel(el: Element): string {
  const tr = el.closest(".transclude")
  if (tr) {
    let prev = tr.previousElementSibling
    while (prev && !/^H[1-6]$/.test(prev.tagName)) prev = prev.previousElementSibling
    if (prev?.textContent) return prev.textContent.trim()
  }
  return el.getAttribute("data-callout") === "exam" ? "Exam questions" : "Flashcards"
}

function collectDecks(root: Element): Deck[] {
  const decks = new Map<string, Deck>()
  const seen = new Set<string>()
  const els = Array.from(root.querySelectorAll(CARD_SELECTOR)).filter(
    (el) => !el.closest(".popover"),
  )
  for (const el of els) {
    const title = el.querySelector(".callout-title-inner")
    const content = el.querySelector(".callout-content")
    if (!title || !content) continue
    const q = title.innerHTML
    const label = deckLabel(el)
    let id = hash(label + "|" + (title.textContent ?? ""))
    while (seen.has(id)) id += "x"
    seen.add(id)
    const card: Card = {
      id,
      q,
      a: content.innerHTML,
      exam: el.getAttribute("data-callout") === "exam",
      deck: label,
    }
    const tr = el.closest(".transclude")
    const anchor = tr ?? el
    if (!decks.has(label)) decks.set(label, { label, cards: [], anchor })
    decks.get(label)!.cards.push(card)
  }
  // Short cards first: on a lecture page the exam questions section comes before the flashcards,
  // but the flashcards are the default thing to study.
  const all = Array.from(decks.values())
  return [...all.filter((d) => !d.cards.every((c) => c.exam)), ...all.filter((d) => d.cards.every((c) => c.exam))]
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag)
  if (cls) e.className = cls
  if (text !== undefined) e.textContent = text
  return e
}

function plural(n: number, word: string) {
  return `${n} ${word}${n === 1 ? "" : "s"}`
}

function pad3(n: number) {
  return String(n).padStart(3, "0")
}

// Small stroke icons, inline so they take the text colour.
const ICONS = {
  close: '<path d="M6 6l12 12M18 6L6 18"/>',
  restart: '<path d="M4 12a8 8 0 1 0 2.4-5.7"/><path d="M4 4v4h4"/>',
  play: '<path d="M7 5l12 7-12 7z" fill="currentColor" stroke="none"/>',
}

function icon(name: keyof typeof ICONS): string {
  return `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="square" aria-hidden="true">${ICONS[name]}</svg>`
}

function iconButton(cls: string, name: keyof typeof ICONS, label: string) {
  const b = el("button", cls)
  b.innerHTML = icon(name)
  b.setAttribute("aria-label", label)
  b.title = label
  return b
}

// "L01b Popper and Lakatos" -> ["01b", "Popper and Lakatos"]. Anything else keeps its full name.
function splitDeckLabel(label: string, index: number): [string, string] {
  const m = label.match(/^L(\d+[a-z]?)\s+(.+)$/i)
  if (m) return [m[1], m[2]]
  return [String(index + 1).padStart(2, "0"), label]
}

// Cards the drill view can show one segment each for. Past this the bar is a plain fill.
const MAX_SEGMENTS = 60

function openSession(cards: Card[], title: string) {
  const byId = new Map(cards.map((c) => [c.id, c]))
  const ids = cards.map((c) => c.id)
  const mixed = new Set(cards.map((c) => c.deck)).size > 1
  const key = `flashcards:${location.pathname}:${hash(ids.slice().sort().join(","))}`

  let state = load(key)
  const valid =
    state &&
    state.ids.length === ids.length &&
    state.ids.every((id) => byId.has(id)) &&
    state.queue.length > 0
  if (!valid) state = { ids, queue: shuffle(ids), done: [], misses: {} }
  let s = state as State
  let revealed = false

  const overlay = el("div", "fc-overlay")
  overlay.setAttribute("role", "dialog")
  overlay.setAttribute("aria-modal", "true")
  overlay.setAttribute("aria-label", title)
  const modal = el("div", "fc-modal")
  overlay.appendChild(modal)

  const top = el("div", "fc-top")
  const close = iconButton("fc-key fc-close", "close", "Close")
  const bar = el("div", "fc-bar")
  const count = el("span", "fc-count")
  const restart = iconButton("fc-key fc-restart", "restart", "Restart")
  top.append(close, bar, count, restart)

  const sub = el("div", "fc-sub")
  const subTitle = el("span", "fc-sub-title", title)
  const tally = el("span", "fc-tally")
  sub.append(subTitle, tally)

  const body = el("div", "fc-body")
  const actions = el("div", "fc-actions")
  const hint = el("div", "fc-hint")
  hint.innerHTML =
    "<kbd>space</kbd> reveal <kbd>1</kbd> missed <kbd>2</kbd> got it <kbd>esc</kbd> close"
  modal.append(top, sub, body, actions, hint)

  const prevOverflow = document.body.style.overflow
  document.body.style.overflow = "hidden"
  document.body.appendChild(overlay)

  function persist() {
    save(key, s)
  }

  function progress() {
    const total = s.ids.length
    count.textContent = `${pad3(s.done.length)}/${pad3(total)}`
    const missTotal = Object.values(s.misses).reduce((a, b) => a + b, 0)
    tally.innerHTML = `got <b>${s.done.length}</b> · miss <b class="fc-tally-miss">${missTotal}</b>`

    bar.replaceChildren()
    if (total <= MAX_SEGMENTS) {
      bar.classList.add("is-segmented")
      // Answered cards first, in the order they were got, black if first try and orange if
      // they needed another go. Then the rest, with the current card outlined.
      for (const id of s.done) {
        bar.appendChild(el("span", (s.misses[id] ?? 0) > 0 ? "fc-seg is-missed" : "fc-seg is-got"))
      }
      for (let i = 0; i < s.queue.length; i++) {
        bar.appendChild(el("span", i === 0 ? "fc-seg is-current" : "fc-seg"))
      }
    } else {
      bar.classList.remove("is-segmented")
      const fill = el("span", "fc-bar-fill")
      fill.style.width = `${(100 * s.done.length) / total}%`
      bar.appendChild(fill)
    }
  }

  function keyButton(cls: string, label: string, onClick: () => void) {
    const b = el("button", `fc-key ${cls}`, label)
    b.addEventListener("click", onClick)
    return b
  }

  function renderCard() {
    progress()
    body.replaceChildren()
    actions.replaceChildren()
    const card = byId.get(s.queue[0])!

    const strip = el("div", "fc-strip")
    const kind = el("span", "fc-kind", card.exam ? "exam" : "recall")
    const side = el("span", revealed ? "fc-side is-back" : "fc-side", revealed ? "back" : "front")
    strip.append(kind, side)

    const inner = el("div", "fc-inner")
    const meta = el("div", "fc-meta")
    // The deck name is already in the header unless this session mixes several decks.
    const parts = mixed ? [card.deck] : []
    const misses = s.misses[card.id] ?? 0
    if (misses > 0) parts.push(`missed ${misses}x`)
    meta.textContent = parts.join(" · ")
    meta.hidden = parts.length === 0

    const q = el("div", "fc-q")
    q.innerHTML = card.q
    inner.append(meta, q)
    if (!revealed) {
      inner.appendChild(
        el(
          "div",
          "fc-prompt",
          card.exam ? "Answer it out loud or on paper first." : "Say the answer before you reveal it.",
        ),
      )
    }
    body.append(strip, inner)
    body.classList.toggle("is-exam", card.exam)

    if (revealed) {
      const a = el("div", "fc-a")
      a.innerHTML = card.a
      inner.appendChild(a)
      actions.append(
        keyButton("fc-miss", "Missed", () => answer(false)),
        keyButton("fc-got fc-key-or", "Got it", () => answer(true)),
      )
    } else {
      actions.append(keyButton("fc-reveal fc-key-ink", "Reveal", reveal))
    }
    inner.scrollTop = 0
    ;(actions.lastElementChild as HTMLElement | null)?.focus({ preventScroll: true })
  }

  function renderDone() {
    progress()
    body.replaceChildren()
    actions.replaceChildren()
    body.classList.remove("is-exam")
    forget(key)

    const missedIds = s.ids.filter((id) => (s.misses[id] ?? 0) > 0)
    const screen = el("div", "fc-done")
    const label = el("span", "fc-done-label", "session complete")
    const grid = el("div", "fc-done-grid")
    const cell = (k: string, v: number, cls = "") => {
      const c = el("div", "fc-done-cell")
      c.append(el("span", "fc-done-k", k), el("span", `fc-done-v ${cls}`, String(v)))
      return c
    }
    grid.append(
      cell("first try", s.ids.length - missedIds.length),
      cell("needed another go", missedIds.length, "is-missed"),
    )
    const line = el(
      "p",
      "fc-done-line",
      missedIds.length === 0
        ? `All ${s.ids.length} right on the first try.`
        : `Done. ${plural(missedIds.length, "card")} of ${s.ids.length} needed another go.`,
    )
    screen.append(label, grid, line)
    body.appendChild(screen)

    if (missedIds.length > 0) {
      const list = el("ul", "fc-missed")
      missedIds
        .sort((x, y) => (s.misses[y] ?? 0) - (s.misses[x] ?? 0))
        .forEach((id) => {
          const li = el("li")
          li.innerHTML = byId.get(id)!.q
          const n = el("span", "fc-missed-n", ` missed ${s.misses[id]}x`)
          li.appendChild(n)
          list.appendChild(li)
        })
      body.appendChild(list)
    }

    if (missedIds.length > 0) {
      actions.append(
        keyButton("fc-miss", "Only the missed", () => {
          teardown()
          openSession(
            missedIds.map((id) => byId.get(id)!),
            title,
          )
        }),
      )
    }
    actions.append(keyButton("fc-got fc-key-or", "All again", startOver))
  }

  function render() {
    if (s.queue.length === 0) renderDone()
    else renderCard()
  }

  function reveal() {
    revealed = true
    render()
  }

  function answer(correct: boolean) {
    const id = s.queue.shift()!
    if (correct) {
      s.done.push(id)
    } else {
      s.misses[id] = (s.misses[id] ?? 0) + 1
      // Back in a few cards: soon enough to fix it this session, late enough not to be a copy.
      const pos = Math.min(s.queue.length, 3 + Math.floor(Math.random() * 3))
      s.queue.splice(pos, 0, id)
    }
    revealed = false
    persist()
    render()
  }

  function startOver() {
    s = { ids, queue: shuffle(ids), done: [], misses: {} }
    revealed = false
    persist()
    render()
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === "Escape") {
      e.preventDefault()
      teardown()
      return
    }
    if (s.queue.length === 0) return
    if (!revealed && (e.key === " " || e.key === "Enter")) {
      e.preventDefault()
      reveal()
    } else if (revealed && (e.key === "1" || e.key === "ArrowLeft")) {
      e.preventDefault()
      answer(false)
    } else if (revealed && (e.key === "2" || e.key === "ArrowRight")) {
      e.preventDefault()
      answer(true)
    }
  }

  function teardown() {
    document.removeEventListener("keydown", onKey, true)
    document.body.style.overflow = prevOverflow
    overlay.remove()
  }

  // Tapping the card face reveals it, so a thumb never has to travel to the button.
  body.addEventListener("click", (e) => {
    if (revealed || s.queue.length === 0) return
    if ((e.target as Element).closest("a")) return
    reveal()
  })
  restart.addEventListener("click", startOver)
  close.addEventListener("click", teardown)
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) teardown()
  })
  document.addEventListener("keydown", onKey, true)
  window.addCleanup(teardown)

  persist()
  render()
}

function screenCell(k: string, v: string, accent = false) {
  const c = el("div", "fc-screen-cell")
  c.append(el("span", "fc-screen-k", k), el("span", accent ? "fc-screen-v is-accent" : "fc-screen-v", v))
  return c
}

function launcher(decks: Deck[]): HTMLElement {
  const box = el("div", "fc-launcher")
  const all = decks.flatMap((d) => d.cards)
  const total = all.length
  const exams = all.filter((c) => c.exam).length

  const screen = el("div", "fc-screen")
  screen.append(screenCell("cards", pad3(total), true))
  if (decks.length > 1) screen.append(screenCell("sets", String(decks.length).padStart(2, "0")))
  screen.append(screenCell("exam-style", pad3(exams)))

  if (decks.length === 1) {
    const go = el("button", "fc-key fc-key-or fc-study")
    go.innerHTML = `${icon("play")}<span>Study</span><span class="fc-key-n">${pad3(total)}</span>`
    go.addEventListener("click", () => openSession(decks[0].cards, decks[0].label))
    const row = el("div", "fc-launcher-row")
    row.append(screen, go)
    box.appendChild(row)
    return box
  }

  box.classList.add("is-multi")
  const pads = el("div", "fc-pads")

  const allPad = el("button", "fc-pad fc-pad-all")
  allPad.innerHTML = `<span class="fc-pad-top"><span>00</span><span>${pad3(total)}</span></span><span class="fc-pad-name">All</span>`
  allPad.setAttribute("aria-label", `Study all ${total} cards`)
  allPad.addEventListener("click", () => openSession(all, "All flashcards"))
  pads.appendChild(allPad)

  const picked = decks.map(() => false)
  const toggles: HTMLButtonElement[] = []
  decks.forEach((d, i) => {
    const [code, name] = splitDeckLabel(d.label, i)
    const padEl = el("div", "fc-pad")
    const toggle = el("button", "fc-pad-toggle")
    toggle.setAttribute("aria-pressed", "false")
    toggle.setAttribute("aria-label", `Add ${d.label} to the mix`)
    const topRow = el("span", "fc-pad-top")
    const id = el("span", "fc-pad-id")
    id.append(el("span", "fc-led"), document.createTextNode(code))
    topRow.append(id, el("span", "fc-pad-count", pad3(d.cards.length)))
    toggle.append(topRow, el("span", "fc-pad-name", name))
    toggle.addEventListener("click", () => {
      picked[i] = !picked[i]
      refresh()
    })
    toggles.push(toggle)

    const play = iconButton("fc-pad-play", "play", `Study ${d.label}`)
    play.addEventListener("click", () => openSession(d.cards, d.label))
    padEl.append(toggle, play)
    pads.appendChild(padEl)
  })

  const foot = el("div", "fc-launcher-foot")
  const go = el("button", "fc-key fc-key-or fc-study fc-study-selected")
  const refresh = () => {
    toggles.forEach((t, i) => {
      t.setAttribute("aria-pressed", String(picked[i]))
      t.parentElement?.classList.toggle("is-on", picked[i])
    })
    const chosen = decks.filter((_, i) => picked[i])
    const n = chosen.reduce((k, d) => k + d.cards.length, 0)
    go.innerHTML = chosen.length
      ? `${icon("play")}<span>Study mix</span><span class="fc-key-n">${pad3(n)}</span>`
      : `<span>Tap pads to mix sets</span>`
    go.disabled = chosen.length === 0
  }
  go.addEventListener("click", () => {
    const chosen = decks.filter((_, i) => picked[i])
    if (chosen.length === 0) return
    openSession(
      chosen.flatMap((d) => d.cards),
      chosen.length === 1 ? chosen[0].label : "Mixed flashcards",
    )
  })
  refresh()
  foot.appendChild(go)
  box.append(screen, pads, foot)
  return box
}

function setupFlashcards() {
  const root = document.querySelector("article")
  if (!root) return
  const decks = collectDecks(root)
  if (decks.length === 0) return

  const first = decks[0].anchor
  let insertBefore: Element = first
  if (first.classList.contains("transclude")) {
    // Course page: put the launcher above the first lecture's heading, not inside it.
    let prev = first.previousElementSibling
    while (prev && !/^H[1-6]$/.test(prev.tagName)) prev = prev.previousElementSibling
    if (prev) insertBefore = prev
  }
  const box = launcher(decks)
  insertBefore.parentElement?.insertBefore(box, insertBefore)
  window.addCleanup(() => box.remove())
}

document.addEventListener("nav", setupFlashcards)
