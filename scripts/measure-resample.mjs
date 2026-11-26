#!/usr/bin/env node
/**
 * Warping error resampling meter (EXPERIMENTAL, docs/MOTION.md §9 "Resampling
 * bias"): drives lab/resample.html over
 * CDP. Synthetic images translate by an exact sub-pixel offset per frame and
 * `WarpingError` judges them through that exact motion, once per resample
 * filter. With nothing flickering, the error left is the filter's own bias
 * (docs/MOTION.md §9, "Resampling bias"); with genuine flicker added, every
 * filter must still read it.
 *
 * Usage: node scripts/measure-resample.mjs [--width 1280] [--height 720]
 *   [--url http://127.0.0.1:5611] [--port 9414] [--adapter swiftshader] [--check]
 *
 * --check fails (exit 1) unless: on smooth detail Catmull-Rom reads ~0 and
 * far below bilinear, on fine detail at least DETAIL_RATIO× below it, on hard
 * edges no worse; every filter reads ~0 on whole-pixel motion and a still
 * image; genuine flicker reads high with both; and Catmull-Rom runs on
 * rgba32float without float32-filterable (bilinear can't, by design).
 *
 * Writes results/resample/resample.json (gitignored).
 */
import { spawn } from 'node:child_process';
import { openSync } from 'node:fs';
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { browserExecutable } from './lib/browser-executable.mjs';
import { CDP } from './lib/cdp-client.mjs';
import { parseCliOrExit } from './lib/cli-flags.mjs';
import { DEFAULT_BENCH_URL, parsePort, removeTempDirectory, resolveServerUrl, spawnVite, stopChild, waitForUrl } from './lib/local-processes.mjs';

const USAGE = `Usage: node scripts/measure-resample.mjs [options]
  --width <px> --height <px>   image size (default 1280x720)
  --url <origin>         lab origin (default ${DEFAULT_BENCH_URL}); started if nothing answers
  --port <n>             Chrome DevTools port (default 9414)
  --adapter swiftshader  software WebGPU (no GPU)
  --dering-ab            also run Catmull-Rom without its deringing clamp
  --check                exit 1 unless the bars hold
Writes results/resample/.`;
const cli = parseCliOrExit(process.argv.slice(2), ['width', 'height', 'url', 'port', 'adapter', 'check', 'dering-ab'], USAGE);
const width = Number(cli.width ?? 1280);
const height = Number(cli.height ?? 720);
const server = resolveServerUrl(cli.url, DEFAULT_BENCH_URL);
const port = parsePort(cli.port, '--port') ?? 9414;
const OUTPUT = resolve(import.meta.dirname, '..', 'results', 'resample');
const FILTERS = ['bilinear', 'catmull-rom'];

//* Bars (conditioned luma)
/** Band-limited detail (zone plate ≤ 0.125 cycles/px): Catmull-Rom's ceiling, and how far below bilinear it reads. */
const SMOOTH_BAR = 2e-4;
const SMOOTH_RATIO = 5;
/** Finer detail, up to 0.35 cycles/px: Catmull-Rom at least this far below bilinear. */
const DETAIL_RATIO = 2;
/** Whole-pixel motion and a still image: no resampling, so every filter reads ~0. */
const ZERO_BAR = 1e-4;
/** Flicker: every filter reads at least this, and Catmull-Rom at least FLICKER_RATIO× its steady twin. */
const FLICKER_FLOOR = 0.01;
const FLICKER_RATIO = 10;

/**
 * Each case: pattern, motion (px/frame, current − previous), extra knobs, and
 * its role in the bars. Hard box-filtered edges (`edges`) aren't band-limited:
 * no interpolator reproduces them under sub-pixel motion, so their only bar is
 * that Catmull-Rom reads no worse than bilinear.
 */
const CASES = [
    { name: 'zoneplate ≤0.125 c/px, (0.37, 0.21)', request: { pattern: 'zoneplate', maxFrequency: 0.125, velocity: [0.37, 0.21] }, role: 'smooth' },
    { name: 'zoneplate ≤0.125 c/px, (0.5, 0)', request: { pattern: 'zoneplate', maxFrequency: 0.125, velocity: [0.5, 0] }, role: 'smooth' },
    { name: 'zoneplate ≤0.125 c/px, (2.3, −1.4)', request: { pattern: 'zoneplate', maxFrequency: 0.125, velocity: [2.3, -1.4] }, role: 'smooth' },
    { name: 'zoneplate ≤0.18 c/px, (0.37, 0.21)', request: { pattern: 'zoneplate', maxFrequency: 0.18, velocity: [0.37, 0.21] }, role: 'detail' },
    { name: 'zoneplate ≤0.25 c/px, (0.37, 0.21)', request: { pattern: 'zoneplate', maxFrequency: 0.25, velocity: [0.37, 0.21] }, role: 'detail' },
    { name: 'zoneplate ≤0.35 c/px, (0.37, 0.21)', request: { pattern: 'zoneplate', maxFrequency: 0.35, velocity: [0.37, 0.21] }, role: 'detail' },
    { name: 'zoneplate ≤0.35 c/px, (0.5, 0)', request: { pattern: 'zoneplate', maxFrequency: 0.35, velocity: [0.5, 0] }, role: 'detail' },
    { name: 'zoneplate ≤0.45 c/px, (0.37, 0.21)', request: { pattern: 'zoneplate', maxFrequency: 0.45, velocity: [0.37, 0.21] }, role: 'edges' },
    { name: 'checker 8 px, (0.37, 0.21)', request: { pattern: 'checker', velocity: [0.37, 0.21] }, role: 'edges' },
    { name: 'checker 8 px, (0.5, 0)', request: { pattern: 'checker', velocity: [0.5, 0] }, role: 'edges' },
    { name: 'edges every 32 px, (0.25, 0)', request: { pattern: 'edges', velocity: [0.25, 0] }, role: 'edges' },
    { name: 'checker 8 px, (1, 0) whole pixel', request: { pattern: 'checker', velocity: [1, 0] }, role: 'zero' },
    { name: 'zoneplate ≤0.35 c/px, still', request: { pattern: 'zoneplate', maxFrequency: 0.35, velocity: [0, 0] }, role: 'zero' },
    { name: 'flicker: zoneplate ≤0.125 + noise ±0.05, (0.37, 0.21)', request: { pattern: 'zoneplate', maxFrequency: 0.125, velocity: [0.37, 0.21], noise: 0.05 }, role: 'flicker', steady: 0 },
    { name: 'flicker: zoneplate ≤0.125, pulse ±5 %, (0.37, 0.21)', request: { pattern: 'zoneplate', maxFrequency: 0.125, velocity: [0.37, 0.21], pulse: 0.05 }, role: 'flicker', steady: 0 },
    { name: 'flicker: checker + noise ±0.05, still', request: { pattern: 'checker', velocity: [0, 0], noise: 0.05 }, role: 'flicker' },
];

const fmt = (value, digits = 5) => (value === null || value === undefined || !Number.isFinite(value) ? '—' : value.toFixed(digits)).padStart(10);

async function main() {
    let vite = null;
    let chrome = null;
    let profile = null;
    let client = null;
    const failures = [];
    try {
        await mkdir(OUTPUT, { recursive: true });
        try {
            await waitForUrl(server.origin, { attempts: 1 });
        } catch {
            vite = spawnVite('lab/vite.config.ts', server);
            await waitForUrl(server.origin, { child: vite });
        }
        profile = await mkdtemp(join(tmpdir(), 'motion-resample-'));
        chrome = spawn(
            browserExecutable(),
            [
                '--headless=new',
                '--enable-unsafe-webgpu',
                `--remote-debugging-port=${port}`,
                `--user-data-dir=${profile}`,
                ...(cli.adapter === 'swiftshader' ? ['--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan'] : []),
                ...(process.env.CI ? ['--no-sandbox'] : []),
                ...(process.env.CHROME_FLAGS ?? '').split(' ').filter(Boolean),
                'about:blank',
            ],
            { stdio: ['ignore', 'ignore', openSync(join(OUTPUT, 'chrome.log'), 'w')] },
        );
        const cdpBase = `http://127.0.0.1:${port}`;
        await waitForUrl(`${cdpBase}/json/version`);
        const created = await fetch(`${cdpBase}/json/new?about:blank`, { method: 'PUT' }).then((response) => response.json());
        client = await CDP.connect(created.webSocketDebuggerUrl);
        await Promise.all([client.send('Page.enable'), client.send('Runtime.enable'), client.send('Log.enable')]);
        await client.send('Page.navigate', { url: `${server.origin}/resample.html` });
        for (let attempt = 0; ; attempt++) {
            if ((await client.evaluate('window.__resampleLab?.ready === true')) === true) break;
            if (attempt > 600) throw new Error(`Timed out waiting for the resample lab.\n${logs(client).join('\n')}`);
            await new Promise((resolveWait) => setTimeout(resolveWait, 100));
        }
        const run = async (request) => {
            const expression = `window.__resampleLab.run(${JSON.stringify({ width, height, ...request })})`;
            const result = await client.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, 5 * 60 * 1000);
            if (result.exceptionDetails) return { error: result.exceptionDetails.exception?.description ?? result.exceptionDetails.text };
            return result.result.value;
        };

        //* Steady, Whole-Pixel And Flicker Cases
        const variants = [...FILTERS.map((resample) => ({ label: resample, resample }))];
        if (cli['dering-ab']) variants.push({ label: 'catmull-rom, no dering', resample: 'catmull-rom', noDering: true });
        const rows = [];
        console.log(`Warping error through exact motion, ${width}×${height}, rgba16float, 15 frames pooled (mean conditioned-luma error)`);
        console.log(`  ${'case'.padEnd(56)}${variants.map((v) => v.label.padStart(Math.max(10, v.label.length + 2))).join('')}`);
        for (const testCase of CASES) {
            const results = {};
            for (const variant of variants) results[variant.label] = await run({ ...testCase.request, resample: variant.resample, noDering: variant.noDering });
            rows.push({ ...testCase, results });
            console.log(`  ${testCase.name.padEnd(56)}${variants.map((v) => fmt(results[v.label].meanError).padStart(Math.max(10, v.label.length + 2))).join('')}`);
            const bilinear = results.bilinear.meanError;
            const cr = results['catmull-rom'].meanError;
            for (const v of variants) if (results[v.label].error) failures.push(`${testCase.name}, ${v.label}: ${results[v.label].error}`);
            if (testCase.role === 'smooth') {
                if (!(cr <= SMOOTH_BAR)) failures.push(`${testCase.name}: catmull-rom reads ${cr} (bar ${SMOOTH_BAR})`);
                if (!(cr * SMOOTH_RATIO <= bilinear)) failures.push(`${testCase.name}: catmull-rom ${cr} not ${SMOOTH_RATIO}× below bilinear ${bilinear}`);
            }
            if (testCase.role === 'detail' && !(cr * DETAIL_RATIO <= bilinear)) failures.push(`${testCase.name}: catmull-rom ${cr} not ${DETAIL_RATIO}× below bilinear ${bilinear}`);
            if (testCase.role === 'edges' && !(cr <= bilinear)) failures.push(`${testCase.name}: catmull-rom ${cr} above bilinear ${bilinear}`);
            if (testCase.role === 'zero') for (const v of variants) if (!(results[v.label].meanError <= ZERO_BAR)) failures.push(`${testCase.name}: ${v.label} reads ${results[v.label].meanError} (bar ${ZERO_BAR})`);
            if (testCase.role === 'flicker') {
                for (const v of FILTERS) if (!(results[v].meanError >= FLICKER_FLOOR)) failures.push(`${testCase.name}: ${v} reads ${results[v].meanError}, under ${FLICKER_FLOOR}`);
                if (testCase.steady !== undefined) {
                    const steady = rows[testCase.steady].results['catmull-rom'].meanError;
                    if (!(cr >= FLICKER_RATIO * steady)) failures.push(`${testCase.name}: catmull-rom ${cr} not ${FLICKER_RATIO}× its steady twin ${steady}`);
                }
            }
        }

        //* Formats
        console.log('\nFormats (zoneplate ≤0.35 c/px, (0.37, 0.21); the device has no float32-filterable):');
        const formats = [];
        for (const format of ['rgba8unorm', 'rgba16float', 'rgba32float']) {
            for (const resample of FILTERS) {
                const result = await run({ pattern: 'zoneplate', maxFrequency: 0.35, velocity: [0.37, 0.21], resample, format, frames: 6 });
                formats.push({ format, resample, meanError: result.meanError ?? null, error: result.error ?? null });
                const text = result.error ? `fails: ${result.error.split('\n')[0].slice(0, 110)}` : fmt(result.meanError).trim();
                console.log(`  ${format.padEnd(12)} ${resample.padEnd(12)} ${text}`);
                const expectedFailure = resample === 'bilinear' && format === 'rgba32float';
                if (!expectedFailure && result.error) failures.push(`${resample} on ${format}: ${result.error}`);
            }
        }

        //* Cost
        // SwiftShader timings mean nothing, and its 2 × 4 × 64 frames take minutes.
        const timing = cli.adapter !== 'swiftshader' && (await client.evaluate('window.__resampleLab.timing'));
        const cost = {};
        if (timing) {
            const response = await client.send(
                'Runtime.evaluate',
                { expression: `window.__resampleLab.cost(${JSON.stringify({ width, height })})`, awaitPromise: true, returnByValue: true },
                5 * 60 * 1000,
            );
            if (response.exceptionDetails) throw new Error(response.exceptionDetails.exception?.description ?? response.exceptionDetails.text);
            const times = response.result.value;
            console.log(`\nGPU time of the warping pass, ${width}×${height}, ${times.bilinear.length} frames each, interleaved (median / p95):`);
            for (const resample of FILTERS) {
                const sorted = [...times[resample]].sort((a, b) => a - b);
                cost[resample] = { median: sorted[Math.floor(sorted.length / 2)], p95: sorted[Math.floor(sorted.length * 0.95)] };
                console.log(`  ${resample.padEnd(12)} ${(cost[resample].median * 1000).toFixed(1)} µs / ${(cost[resample].p95 * 1000).toFixed(1)} µs`);
            }
        }

        const strip = (result) => ({ meanError: result.meanError, scored: result.scored, error: result.error, perFrame: result.frames?.map((frame) => frame.meanError) });
        await writeFile(
            join(OUTPUT, 'resample.json'),
            `${JSON.stringify({ width, height, rows: rows.map((row) => ({ ...row, results: Object.fromEntries(Object.entries(row.results).map(([key, value]) => [key, strip(value)])) })), formats, cost }, null, 2)}\n`,
        );
        const records = logs(client);
        await writeFile(join(OUTPUT, 'browser-log.txt'), `${records.join('\n')}\n`);
    } finally {
        client?.close();
        await stopChild(chrome);
        await removeTempDirectory(profile, 'Chrome profile');
        await stopChild(vite);
    }
    if (cli.check) {
        if (failures.length) {
            console.error(`\nresample bars: ${failures.length} failed\n  ${failures.join('\n  ')}`);
            process.exitCode = 1;
        } else console.log('\nresample bars: every bar holds');
    } else if (failures.length) console.warn(`\n(${failures.length} bar(s) would fail with --check)\n  ${failures.join('\n  ')}`);
    console.log(`\nartifacts: ${OUTPUT}`);
}

function logs(client) {
    return client.events
        .filter((event) => event.method === 'Log.entryAdded' || event.method === 'Runtime.exceptionThrown' || event.method === 'Runtime.consoleAPICalled')
        .map((event) => {
            if (event.method === 'Log.entryAdded') return `[${event.params.entry.level}] ${event.params.entry.text}`;
            if (event.method === 'Runtime.exceptionThrown') return `[exception] ${event.params.exceptionDetails.exception?.description ?? event.params.exceptionDetails.text}`;
            return `[console.${event.params.type}] ${event.params.args.map((arg) => arg.value ?? arg.description).join(' ')}`;
        });
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
