#!/usr/bin/env node
/**
 * Packs the package and checks the tarball a consumer would install: every
 * `exports` target exists with its types, nothing outside `dist/` and the
 * contract ships, no import names another package of ours or a relative path
 * that escapes `dist/`, each engine is imported only from its own subpath
 * (`three` from `/three`, `@pixi/3d` and `pixi.js` from `/pixi`), and the
 * engine-free entries import in plain Node.
 * Runs nothing on the GPU.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const ROOT = resolve(import.meta.dirname, '..');
const work = mkdtempSync(join(tmpdir(), 'motion-vectors-pack-'));
const problems = [];
try {
    const [packed] = JSON.parse(execFileSync('npm', ['pack', '--json', '--pack-destination', work], { cwd: ROOT, encoding: 'utf8' }));
    execFileSync('tar', ['xzf', join(work, packed.filename), '-C', work]);
    const root = join(work, 'package');
    const manifest = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));

    for (const file of packed.files.map((entry) => entry.path))
        if (!/^(dist\/|CONTRACT\.md$|CHANGELOG\.md$|README\.md$|LICENSE$|package\.json$)/.test(file)) problems.push(`unexpected file in the tarball: ${file}`);
    for (const [subpath, target] of Object.entries(manifest.exports)) {
        const targets = typeof target === 'string' ? [target] : Object.values(target);
        for (const file of targets) if (!existsSync(join(root, file))) problems.push(`exports["${subpath}"] → ${file} is missing`);
    }

    const walk = (dir) => readdirSync(dir).flatMap((name) => (statSync(join(dir, name)).isDirectory() ? walk(join(dir, name)) : [join(dir, name)]));
    for (const file of walk(join(root, 'dist')).filter((path) => /\.(js|d\.ts)$/.test(path))) {
        const source = readFileSync(file, 'utf8');
        for (const [, specifier] of source.matchAll(/(?:from|import)\s*\(?\s*['"]([^'"]+)['"]/g)) {
            if (/^@pmndrs\//.test(specifier)) problems.push(`${relative(root, file)} imports ${specifier}`);
            if (specifier.startsWith('.') && !resolve(file, '..', specifier).startsWith(join(root, 'dist'))) problems.push(`${relative(root, file)} escapes dist: ${specifier}`);
            if (specifier.startsWith('.') && file.endsWith('.js') && !specifier.endsWith('.js')) problems.push(`${relative(root, file)}: extensionless ${specifier}`);
            // Each engine only from its own subpath, so one engine never drags in another.
            const where = relative(join(root, 'dist'), file).split(/[\\/]/)[0];
            const allowed = where === 'three' ? /^three(\/|$)/ : where === 'pixi' ? /^(@pixi\/3d|pixi\.js)(\/|$)/ : null;
            if (!specifier.startsWith('.') && !allowed?.test(specifier)) problems.push(`${relative(root, file)} imports ${specifier} outside its engine's subpath`);
        }
    }

    // The engine-free entries must load without three or a GPU.
    for (const entry of ['.', './webgpu']) {
        const file = join(root, manifest.exports[entry].default);
        const module = await import(pathToFileURL(file).href);
        if (Object.keys(module).length === 0) problems.push(`${entry} exports nothing`);
    }
    console.log(`${packed.filename}: ${packed.files.length} files, ${packed.size} bytes packed`);
} finally {
    rmSync(work, { recursive: true, force: true });
}
if (problems.length) {
    console.error(`verify-pack: ${problems.length} problem(s)\n  ${problems.join('\n  ')}`);
    process.exitCode = 1;
} else console.log('verify-pack: ok');
