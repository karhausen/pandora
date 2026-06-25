# MVP 24.9.1 – Review Scheduler GUI Theme Fix

Fixes the Review Scheduler web page so it consistently uses Pandora's dark theme and shared badge navigation style.

## Changes

- Review Scheduler page now defines the same dark base layout as Operations and Night Review pages.
- Cards, summary panels, inputs, buttons, list items and preformatted output use the shared color variables.
- The page no longer falls back to the browser's white default background.

## Verification

- `/review-scheduler` uses `web/shared.css` and `web/review-scheduler.css`.
- CSS contains explicit `body`, `.topbar`, `.card`, `.summary`, `.item`, `input`, `button`, and `pre` dark-theme rules.
