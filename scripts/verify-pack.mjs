#!/usr/bin/env node
/**
 * Packs the package and checks the tarball a consumer would install: every
 * `exports` target exists with its types, nothing outside `dist/` and the
 * contract ships, no import names another package of ours or a relative path
 * that escapes `dist/`, each engine is imported only from its own subpath
 * (`three` from `/three`, `@pixi/3d` and `pixi.js` from `/pixi`), and the
 * engine-free entries import in plain Node. Then a fresh consumer installs the
 * tarball with each engine at the versions this repo develops against and
 * typechecks every entry with `skipLibCheck` off (`--no-consumer` skips it; it
 * needs the npm registry).
 * Runs nothing on the GPU.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
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

    // A consumer's typecheck: our declarations must compile with nothing but the
    // package, its dependencies and the engine the subpath names installed.
    if (!process.argv.includes('--no-consumer')) {
        const consumer = join(work, 'consumer');
        mkdirSync(consumer);
        writeFileSync(join(consumer, 'package.json'), JSON.stringify({ name: 'consumer', private: true, type: 'module' }));
        const dev = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8')).devDependencies;
        const engines = ['three', '@types/three', 'pixi.js', '@pixi/3d', 'typescript'].map((name) => `${name}@${dev[name]}`);
        execFileSync('npm', ['install', '--no-audit', '--no-fund', '--loglevel=error', join(work, packed.filename), ...engines], { cwd: consumer, stdio: 'pipe' });
        // One program per subpath: an engine's own declarations can pull in globals
        // (pixi.js references @webgpu/types), which would hide a gap in ours.
        const programs = {
            'webgpu.ts': [
                "import { MOTION_SCALE, MotionFlag } from '@pmndrs/motion-vectors';",
                "import { CutDetector, MotionAnalyzer, MotionFromDepth, WarpingError, type TextureAllocator } from '@pmndrs/motion-vectors/webgpu';",
                'export const entries = [MOTION_SCALE, MotionFlag.Exact, CutDetector, MotionAnalyzer, MotionFromDepth, WarpingError];',
                'export type Allocator = TextureAllocator;',
            ],
            'three.ts': [
                "import { MotionFromDepthPass, MotionVectors } from '@pmndrs/motion-vectors/three';",
                'export const entries = [MotionFromDepthPass, MotionVectors];',
            ],
            'pixi.ts': [
                "import { MotionVectors, motionOutput } from '@pmndrs/motion-vectors/pixi';",
                'export const entries = [MotionVectors, motionOutput];',
            ],
        };
        const tsc = join(consumer, 'node_modules', 'typescript', 'bin', 'tsc');
        const ours = [];
        for (const [name, lines] of Object.entries(programs)) {
            writeFileSync(join(consumer, name), `${lines.join('\n')}\n`);
            // `types: []`: no ambient @types, so only what our declarations reference is in scope.
            const config = `tsconfig.${name}.json`;
            const compilerOptions = { noEmit: true, strict: true, module: 'nodenext', moduleResolution: 'nodenext', target: 'es2022', lib: ['es2022', 'dom'], types: [] };
            writeFileSync(join(consumer, config), JSON.stringify({ compilerOptions, files: [name] }));
            let output = '';
            try {
                execFileSync(process.execPath, [tsc, '-p', config], { cwd: consumer, encoding: 'utf8', stdio: 'pipe' });
            } catch (error) {
                output = `${error.stdout ?? ''}${error.stderr ?? ''}`;
            }
            // Every error counts except those inside an engine's own declarations.
            const errors = output.split('\n').filter((line) => /error TS\d+/.test(line));
            ours.push(...errors.filter((line) => !/^node_modules\//.test(line) || /^node_modules\/@pmndrs\/motion-vectors\//.test(line)).map((line) => `${name}: ${line}`));
        }
        for (const line of ours) problems.push(`consumer typecheck: ${line}`);
        if (!ours.length) console.log(`consumer typecheck: ok (${engines.join(', ')})`);
    }
} finally {
    rmSync(work, { recursive: true, force: true });
}
if (problems.length) {
    console.error(`verify-pack: ${problems.length} problem(s)\n  ${problems.join('\n  ')}`);
    process.exitCode = 1;
} else console.log('verify-pack: ok');
