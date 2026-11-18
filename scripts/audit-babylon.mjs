#!/usr/bin/env node
/**
 * Babylon velocity audit (EXPERIMENTAL, docs/PLAN.md Phase 4): drives
 * lab/babylon.html over CDP and reports, per deformers object, the mean
 * endpoint error of Babylon's own linear velocity against exact motion
 * (render px). Findings: docs/MOTION.md §11.
 *
 * The rigid box and the floor under a still or translating camera are the
 * harness's controls: Babylon's velocity is exact on them at one render per
 * frame, so a non-zero error there means the oracle or the readback is wrong,
 * and the run fails. Every other row is a finding, not a bar.
 *
 * Usage: node scripts/audit-babylon.mjs [--width 1280] [--height 720] [--port 9413]
 *   [--url http://127.0.0.1:5611] [--adapter swiftshader] [--end 239]
 */
import { spawn } from 'node:child_process';
import { openSync, readFileSync } from 'node:fs';
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { browserExecutable } from './lib/browser-executable.mjs';
import { CDP } from './lib/cdp-client.mjs';
import { parseCliOrExit } from './lib/cli-flags.mjs';
import { DEFAULT_BENCH_URL, parsePort, removeTempDirectory, resolveServerUrl, spawnVite, stopChild, waitForUrl } from './lib/local-processes.mjs';

const USAGE = 'Usage: node scripts/audit-babylon.mjs [--width 1280] [--height 720] [--port 9413] [--url http://127.0.0.1:5611] [--adapter swiftshader] [--end 239]';
const cli = parseCliOrExit(process.argv.slice(2), ['width', 'height', 'port', 'url', 'adapter', 'end'], USAGE);
const width = Number(cli.width ?? 1280);
const height = Number(cli.height ?? 720);
const port = parsePort(cli.port, '--port') ?? 9413;
const server = resolveServerUrl(cli.url, DEFAULT_BENCH_URL);
const end = Number(cli.end ?? 239);
const OUTPUT = resolve(import.meta.dirname, '..', 'results', 'babylon');
const CHROME_LOG = join(OUTPUT, 'chrome.log');
/** The controls' bar (CONTRACT.md's exact): mean error in render px. */
const EXACT = 0.01;

/** Each case: what runs, and whether it is a control (must read exact). */
const CASES = [
    { name: 'floor, still camera', request: { object: 'floor' }, control: true },
    { name: 'floor, camera moving', request: { object: 'floor', cameraMove: true }, control: true },
    { name: 'floor, camera moving, 2 renders', request: { object: 'floor', cameraMove: true, rendersPerFrame: 2 } },
    { name: 'rigid box', request: { object: 'rigid' }, control: true },
    { name: 'rigid box, 2 renders', request: { object: 'rigid', rendersPerFrame: 2 } },
    { name: 'rigid box, hidden 60–89', request: { object: 'rigid', hide: [60, 89] } },
    { name: 'static displacement', request: { object: 'displaced' } },
    { name: 'animated displacement', request: { object: 'wave' } },
    { name: 'morph target', request: { object: 'morph' } },
    { name: 'leaves and re-enters', request: { object: 'reentry' } },
    { name: 'skinned limb (bone texture)', request: { object: 'skinned' } },
    { name: 'skinned limb, bone uniforms', request: { object: 'skinned', boneUniforms: true } },
    { name: 'skinned limb, bone uniforms, 2 renders', request: { object: 'skinned', boneUniforms: true, rendersPerFrame: 2 } },
];

const fmt = (value, digits = 3) => (value === null || value === undefined || Number.isNaN(value) ? '—' : value.toFixed(digits)).padStart(9);

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
        profile = await mkdtemp(join(tmpdir(), 'motion-babylon-'));
        chrome = spawn(
            browserExecutable(),
            [
                '--headless=new',
                '--enable-unsafe-webgpu',
                `--remote-debugging-port=${port}`,
                `--user-data-dir=${profile}`,
                `--window-size=${width},${height}`,
                '--force-device-scale-factor=1',
                ...(cli.adapter === 'swiftshader' ? ['--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan'] : []),
                ...(process.env.CI ? ['--no-sandbox'] : []),
                ...(process.env.CHROME_FLAGS ?? '').split(' ').filter(Boolean),
                'about:blank',
            ],
            { stdio: ['ignore', 'ignore', openSync(CHROME_LOG, 'w')] },
        );
        const cdpBase = `http://127.0.0.1:${port}`;
        await waitForUrl(`${cdpBase}/json/version`);
        const created = await fetch(`${cdpBase}/json/new?about:blank`, { method: 'PUT' }).then((response) => response.json());
        client = await CDP.connect(created.webSocketDebuggerUrl);
        await Promise.all([client.send('Page.enable'), client.send('Runtime.enable'), client.send('Log.enable')]);
        await client.send('Page.navigate', { url: `${server.origin}/babylon.html?width=${width}&height=${height}&present=0` });
        for (let attempt = 0; ; attempt++) {
            if ((await client.evaluate('window.__babylonAudit?.ready === true')) === true) break;
            if (attempt > 600) throw new Error(`Timed out waiting for the Babylon audit page.\n${logs(client).join('\n')}`);
            await new Promise((resolveWait) => setTimeout(resolveWait, 100));
        }
        const version = await client.evaluate('window.__babylonAudit.babylon');
        const results = [];
        for (const testCase of CASES) {
            const expression = `window.__babylonAudit.run(${JSON.stringify({ ...testCase.request, end })})`;
            const result = await client.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, 10 * 60 * 1000);
            if (result.exceptionDetails) throw new Error(`${testCase.name}: ${result.exceptionDetails.exception?.description ?? result.exceptionDetails.text}\n${logs(client).join('\n')}`);
            results.push({ ...testCase, frames: result.result.value.frames, defines: result.result.value.defines });
        }
        await writeFile(join(OUTPUT, 'audit.json'), `${JSON.stringify({ babylon: version, width, height, end, results }, null, 2)}\n`);

        console.log(`Babylon.js ${version} velocity audit, ${width}×${height}, frames 1–${end}: mean endpoint error in render px against exact motion`);
        console.log(`  ${'case'.padEnd(40)} ${'frames'.padStart(6)} ${'motion'.padStart(9)} ${'reported'.padStart(9)} ${'error'.padStart(9)} ${'max'.padStart(9)}   worst frame`);
        for (const row of results) {
            const seen = row.frames.filter((frame) => frame.pixels > 0);
            const mean = (key) => seen.reduce((sum, frame) => sum + frame[key], 0) / seen.length;
            const worst = seen.reduce((a, b) => (b.babylon > a.babylon ? b : a), seen[0]);
            const max = Math.max(...seen.map((frame) => frame.max));
            const babylon = mean('babylon');
            console.log(
                `  ${row.name.padEnd(40)} ${String(seen.length).padStart(6)} ${fmt(mean('motion'))} ${fmt(mean('reported'))} ${fmt(babylon)} ${fmt(max, 2)}   ${worst.frame} (${worst.babylon.toFixed(2)} px${worst.reentered ? ', shown again' : ''})`,
            );
            for (const frame of row.frames.filter((f) => f.reentered))
                console.log(`  ${''.padEnd(40)} shown again, frame ${frame.frame}: ${frame.babylon.toFixed(2)} px (motion ${frame.motion.toFixed(2)} px)`);
            if (row.control && !(babylon <= EXACT)) failures.push(`${row.name}: control reads ${babylon.toFixed(4)} px (bar ${EXACT}); the harness is wrong`);
            if (seen.length === 0) failures.push(`${row.name}: no pixels measured`);
        }
        const records = logs(client);
        await writeFile(join(OUTPUT, 'browser-log.txt'), `${records.join('\n')}\n`);
        if (records.some((line) => /error|exception|validation|invalid/i.test(line))) {
            console.warn(`\nbrowser log:\n${records.join('\n')}`);
            failures.push('browser log has errors');
        }
    } finally {
        client?.close();
        await stopChild(chrome);
        await removeTempDirectory(profile, 'Chrome profile');
        await stopChild(vite);
    }
    if (failures.length) {
        console.error(`\nBabylon audit: ${failures.length} failure(s)\n  ${failures.join('\n  ')}`);
        process.exitCode = 1;
    } else console.log(`\nBabylon audit: controls exact; raw frames in ${join(OUTPUT, 'audit.json')}`);
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
    try {
        const tail = readFileSync(CHROME_LOG, 'utf8').trim().split('\n').slice(-40).join('\n');
        if (tail) console.error(`\nchrome.log (last 40 lines):\n${tail}`);
    } catch {
        // no Chrome log yet
    }
    process.exitCode = 1;
});
