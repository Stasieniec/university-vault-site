// @ts-ignore
import flashcardsScript from "./scripts/flashcards.inline"
import styles from "./styles/flashcards.scss"
import { QuartzComponent, QuartzComponentConstructor } from "./types"

// Renders nothing on its own. The script finds `[!card]` and `[!exam]` callouts on the page
// and adds a study mode on top of them. Progress lives in sessionStorage only, per browser tab,
// because the site is shared and nobody's progress should be stored anywhere else.
const Flashcards: QuartzComponent = () => null

Flashcards.afterDOMLoaded = flashcardsScript
Flashcards.css = styles

export default (() => Flashcards) satisfies QuartzComponentConstructor
