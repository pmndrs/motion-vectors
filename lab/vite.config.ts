import { resolve } from 'path';
import { defineConfig } from 'vite';

// The motion lab: the package's GPU acceptance harness (docs/PLAN.md §4).
export default defineConfig({
    root: __dirname,
    // Own dep-optimizer cache, so it can run next to the bench and examples.
    cacheDir: resolve(__dirname, '../node_modules/.vite-lab'),
    // Top-level await (renderer.init) needs a modern target.
    build: { target: 'esnext' },
    server: { port: 5611, open: false },
});
