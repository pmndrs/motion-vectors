import { resolve } from 'path';
import { defineConfig } from 'vite';

// The example gallery. Examples import the package by its public name; the
// alias points it at src/, so they run against the working tree, no build.
export default defineConfig({
    root: __dirname,
    cacheDir: resolve(__dirname, '../node_modules/.vite-examples'),
    resolve: {
        alias: [
            { find: /^@pmndrs\/motion-vectors\/three$/, replacement: resolve(__dirname, '../src/three/index.ts') },
            { find: /^@pmndrs\/motion-vectors\/pixi$/, replacement: resolve(__dirname, '../src/pixi/index.ts') },
            { find: /^@pmndrs\/motion-vectors\/webgpu$/, replacement: resolve(__dirname, '../src/webgpu/index.ts') },
            { find: /^@pmndrs\/motion-vectors$/, replacement: resolve(__dirname, '../src/index.ts') },
        ],
    },
    // Top-level await (renderer.init) needs a modern target.
    build: {
        target: 'esnext',
        rollupOptions: {
            input: {
                index: resolve(__dirname, 'index.html'),
                'three-compare': resolve(__dirname, 'three-compare/index.html'),
                'pixi-compare': resolve(__dirname, 'pixi-compare/index.html'),
            },
        },
    },
    server: { port: 5612, open: false },
});
