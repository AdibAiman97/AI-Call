import { onMounted, onUnmounted } from 'vue'

type HotkeyCallback = (event: KeyboardEvent) => void;

/**
 * A Vue composable for handling keyboard shortcuts (hotkeys).
 *
 * @param key The key to listen for (e.g., 'k', 'Enter', 'ArrowUp').
 * @param callback The function to execute when the hotkey is pressed.
 * @param modifiers Optional modifiers (ctrl, shift, alt).
 */
export function useHotkey(
  key: string,
  callback: HotkeyCallback,
  modifiers: { ctrl?: boolean; shift?: boolean; alt?: boolean } = {}
) {
  const handleKeyDown = (event: KeyboardEvent) => {
    const { ctrl = false, shift = false, alt = false } = modifiers

    // Check for modifier keys (metaKey is for Command key on macOS)
    const ctrlMatch = ctrl ? event.ctrlKey || event.metaKey : true
    const shiftMatch = shift ? event.shiftKey : true
    const altMatch = alt ? event.altKey : true

    if (
      event.key.toLowerCase() === key.toLowerCase() &&
      ctrlMatch &&
      shiftMatch &&
      altMatch
    ) {
      event.preventDefault()
      callback(event)
    }
  }

  onMounted(() => {
    window.addEventListener('keydown', handleKeyDown)
  })

  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeyDown)
  })
}
