/// <reference types="vite/client" />

// Vite's client types declare CSS and asset modules. Without them `tsc`
// rejects side-effect imports such as `import './index.css'` (TS2882). A
// reference kept next to the sources does that without narrowing the global
// `compilerOptions.types` list.
