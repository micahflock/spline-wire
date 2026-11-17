/// <reference types="vite/client" />

// What the pages read from import.meta.env besides Vite's own.
interface ImportMetaEnv {
  /** The relay's address for this build (src/relay/config.ts); unset in development. */
  readonly VITE_RELAY_URL?: string;
  /** Set in vite.config.ts: the site's version and the CI run that built it ("dev" locally). */
  readonly APP_VERSION: string;
  readonly BUILD_ID: string;
}
