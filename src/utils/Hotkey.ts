import { onMounted, onUnmounted } from 'vue'

type HotkeyCallback = (event: KeyboardEvent) => void;

/**
 * A Vue composable for handling keyboard shortcuts (hotkeys).
 *
 * @param key The key to listen for (e.g., 'k', 'Enter', 'ArrowUp').
 * @param callback The function to execute when the hotkey is pressed.
 * @param modifiers Optional modifiers (ctrl, shift, alt, command). `command` maps to the meta key.
 */
export function useHotkey(
  key: string,
  callback: HotkeyCallback,
  modifiers: { ctrl?: boolean; shift?: boolean; alt?: boolean; command?: boolean } = {},
) {
  const handleKeyDown = (event: KeyboardEvent) => {
    const {
      ctrl: wantCtrl = false,
      shift: wantShift = false,
      alt: wantAlt = false,
      command: wantCommand = false,
    } = modifiers

    if (
      event.key.toLowerCase() === key.toLowerCase() &&
      event.ctrlKey === wantCtrl &&
      event.shiftKey === wantShift &&
      event.altKey === wantAlt &&
      event.metaKey === wantCommand
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
