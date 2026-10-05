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
// heading. On a lecture page there is no transclusion and the deck is the note itself.
function deckLabel(el: Element): string {
  const tr = el.closest(".transclude")
  if (tr) {
    let prev = tr.previousElementSibling
    while (prev && !/^H[1-6]$/.test(prev.tagName)) prev = prev.previousElementSibling
    if (prev?.textContent) return prev.textContent.trim()
  }
  return document.querySelector(".article-title")?.textContent?.trim() ?? "Flashcards"
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
  return Array.from(decks.values())
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

function openSession(cards: Card[], title: string) {
  const byId = new Map(cards.map((c) => [c.id, c]))
  const ids = cards.map((c) => c.id)
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
  const count = el("span", "fc-count")
  const restart = el("button", "fc-restart", "Restart")
  const close = el("button", "fc-close", "Close")
  top.append(count, restart, close)

  const bar = el("div", "fc-bar")
  const fill = el("div", "fc-bar-fill")
  bar.appendChild(fill)

  const body = el("div", "fc-body")
  const actions = el("div", "fc-actions")
  const hint = el("div", "fc-hint", "Space: show answer. 1: missed it. 2: got it. Esc: close.")
  modal.append(top, bar, body, actions, hint)

  const prevOverflow = document.body.style.overflow
  document.body.style.overflow = "hidden"
  document.body.appendChild(overlay)

  function persist() {
    save(key, s)
  }

  function progress() {
    const total = s.ids.length
    count.textContent = `${s.done.length} of ${total} done`
    fill.style.width = `${(100 * s.done.length) / total}%`
  }

  function button(cls: string, label: string, onClick: () => void) {
    const b = el("button", cls, label)
    b.addEventListener("click", onClick)
    return b
  }

  function renderCard() {
    progress()
    body.replaceChildren()
    actions.replaceChildren()
    const card = byId.get(s.queue[0])!

    const meta = el("div", "fc-meta")
    const parts = [card.deck]
    if (card.exam) parts.push("Exam question: answer it out loud or on paper first")
    const misses = s.misses[card.id] ?? 0
    if (misses > 0) parts.push(`missed ${misses}x this session`)
    meta.textContent = parts.join(" · ")

    const q = el("div", "fc-q")
    q.innerHTML = card.q
    body.append(meta, q)
    if (card.exam) body.classList.add("is-exam")
    else body.classList.remove("is-exam")

    if (revealed) {
      const a = el("div", "fc-a")
      a.innerHTML = card.a
      body.appendChild(a)
      actions.append(
        button("fc-miss", "Missed it", () => answer(false)),
        button("fc-got", "Got it", () => answer(true)),
      )
    } else {
      actions.append(button("fc-reveal", "Show answer", reveal))
    }
    body.scrollTop = 0
    ;(actions.lastElementChild as HTMLElement | null)?.focus({ preventScroll: true })
  }

  function renderDone() {
    progress()
    body.replaceChildren()
    actions.replaceChildren()
    body.classList.remove("is-exam")
    forget(key)

    const missedIds = s.ids.filter((id) => (s.misses[id] ?? 0) > 0)
    const head = el("div", "fc-q")
    head.textContent =
      missedIds.length === 0
        ? `All ${s.ids.length} right on the first try.`
        : `Done. ${plural(missedIds.length, "card")} of ${s.ids.length} needed another go.`
    body.appendChild(head)

    if (missedIds.length > 0) {
      const list = el("ul", "fc-missed")
      missedIds
        .sort((x, y) => (s.misses[y] ?? 0) - (s.misses[x] ?? 0))
        .forEach((id) => {
          const li = el("li")
          li.innerHTML = byId.get(id)!.q
          const n = el("span", "fc-missed-n", ` (missed ${s.misses[id]}x)`)
          li.appendChild(n)
          list.appendChild(li)
        })
      body.appendChild(list)
    }

    if (missedIds.length > 0) {
      actions.append(
        button("fc-miss", "Only the missed ones", () => {
          teardown()
          openSession(
            missedIds.map((id) => byId.get(id)!),
            title,
          )
        }),
      )
    }
    actions.append(button("fc-got", "All again", startOver))
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

function launcher(decks: Deck[]): HTMLElement {
  const box = el("div", "fc-launcher")
  const total = decks.reduce((n, d) => n + d.cards.length, 0)

  if (decks.length === 1) {
    const label = el("span", "fc-launcher-label", plural(total, "flashcard"))
    const go = el("button", "fc-study", "Study")
    go.addEventListener("click", () => openSession(decks[0].cards, decks[0].label))
    box.append(label, go)
    return box
  }

  box.classList.add("is-multi")
  const head = el("div", "fc-launcher-head")
  const label = el("span", "fc-launcher-label", `${plural(total, "flashcard")} in ${decks.length} sets`)
  const all = el("button", "fc-study", `Study all ${total}`)
  all.addEventListener("click", () => openSession(decks.flatMap((d) => d.cards), "All flashcards"))
  head.append(label, all)

  const list = el("div", "fc-deck-list")
  const boxes: HTMLInputElement[] = []
  for (const d of decks) {
    const row = el("div", "fc-deck")
    const pick = el("label", "fc-deck-pick")
    const cb = el("input")
    cb.type = "checkbox"
    boxes.push(cb)
    const name = el("span", "fc-deck-name", d.label)
    const n = el("span", "fc-deck-count", String(d.cards.length))
    pick.append(cb, name)
    const one = el("button", "fc-deck-study", "Study")
    one.addEventListener("click", () => openSession(d.cards, d.label))
    row.append(pick, n, one)
    list.appendChild(row)
  }

  const foot = el("div", "fc-launcher-foot")
  const go = el("button", "fc-study fc-study-selected", "Study ticked")
  const refresh = () => {
    const chosen = decks.filter((_, i) => boxes[i].checked)
    const n = chosen.reduce((k, d) => k + d.cards.length, 0)
    go.textContent = chosen.length ? `Study ticked (${n})` : "Tick lectures to mix them"
    go.disabled = chosen.length === 0
  }
  boxes.forEach((cb) => cb.addEventListener("change", refresh))
  go.addEventListener("click", () => {
    const chosen = decks.filter((_, i) => boxes[i].checked)
    if (chosen.length === 0) return
    openSession(
      chosen.flatMap((d) => d.cards),
      chosen.length === 1 ? chosen[0].label : "Mixed flashcards",
    )
  })
  refresh()
  foot.appendChild(go)
  box.append(head, list, foot)
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
