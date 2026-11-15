#!/usr/bin/env node
/**
 * pixi-3d adapter check (EXPERIMENTAL): drives examples/pixi-compare in check
 * mode over CDP and compares `MotionVectors` and pixi-3d's own velocity against
 * analytic motion. The check scene has a backdrop and a plane facing a camera
 * that only translates, so every point of each moves by one NDC delta,
 * projected on the CPU from the camera's matrices. Exact means a mean error
 * under 0.01 render px (CONTRACT.md) plus one f16 step of the stored value.
 *
 * Usage: node scripts/check-pixi.mjs [--width 1280] [--height 720] [--port 9412]
 *   [--url http://127.0.0.1:5613] [--adapter swiftshader]
 */
import { spawn } from 'node:child_process';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { browserExecutable } from './lib/browser-executable.mjs';
import { CDP } from './lib/cdp-client.mjs';
import { parseCliOrExit } from './lib/cli-flags.mjs';
import { parsePort, removeTempDirectory, resolveServerUrl, spawnVite, stopChild, waitForUrl } from './lib/local-processes.mjs';

const USAGE = 'Usage: node scripts/check-pixi.mjs [--width 1280] [--height 720] [--port 9412] [--url http://127.0.0.1:5613] [--adapter swiftshader]';
const cli = parseCliOrExit(process.argv.slice(2), ['width', 'height', 'port', 'url', 'adapter'], USAGE);
const width = Number(cli.width ?? 1280);
const height = Number(cli.height ?? 720);
const port = parsePort(cli.port, '--port') ?? 9412;
const server = resolveServerUrl(cli.url, 'http://127.0.0.1:5613');
const EXACT = 0.01;
/** Cases that show a pixi-3d limit rather than ours (docs/PIXI.md, upstream asks). */
const KNOWN_LIMITS = new Set(['camera moves, 2 renders']);

/** Spacing of f16 values around `value` (normal range). */
const f16Step = (value) => (value === 0 ? 0 : 2 ** (Math.floor(Math.log2(Math.abs(value))) - 10));
const fmt = (value, digits = 3) => (value === null || value === undefined ? '—' : value.toFixed(digits)).padStart(8);

async function main() {
    let vite = null;
    let chrome = null;
    let profile = null;
    let client = null;
    const failures = [];
    try {
        try {
            await waitForUrl(server.origin, { attempts: 1 });
        } catch {
            vite = spawnVite('examples/vite.config.ts', server);
            await waitForUrl(server.origin, { child: vite });
        }
        profile = await mkdtemp(join(tmpdir(), 'motion-pixi-'));
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
                'about:blank',
            ],
            { stdio: 'ignore' },
        );
        const cdpBase = `http://127.0.0.1:${port}`;
        await waitForUrl(`${cdpBase}/json/version`);
        const created = await fetch(`${cdpBase}/json/new?about:blank`, { method: 'PUT' }).then((response) => response.json());
        client = await CDP.connect(created.webSocketDebuggerUrl);
        await Promise.all([client.send('Page.enable'), client.send('Runtime.enable'), client.send('Log.enable')]);
        await client.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
        await client.send('Page.navigate', { url: `${server.origin}/pixi-compare/index.html?check=1` });
        for (let attempt = 0; ; attempt++) {
            if ((await client.evaluate('window.__pixiMotion?.ready === true')) === true) break;
            if (attempt > 300) throw new Error(`Timed out waiting for the pixi check page.\n${logs(client).join('\n')}`);
            await new Promise((resolveWait) => setTimeout(resolveWait, 100));
        }
        const result = await client.send('Runtime.evaluate', { expression: 'window.__pixiMotion.check()', awaitPromise: true, returnByValue: true }, 5 * 60 * 1000);
        if (result.exceptionDetails) throw new Error(`${result.exceptionDetails.exception?.description ?? result.exceptionDetails.text}\n${logs(client).join('\n')}`);
        const cases = result.result.value;
        const size = await client.evaluate('window.__pixiMotion.size');
        console.log(`pixi-3d check, ${size[0]}×${size[1]}: mean / max endpoint error in render px against analytic motion`);
        console.log(`  ${'case'.padEnd(24)} ${'region'.padEnd(9)} ${'motion px'.padStart(9)} ${'ours mean'.padStart(9)} ${'ours max'.padStart(8)} ${'pixi mean'.padStart(9)} ${'pixi max'.padStart(8)}`);
        for (const row of cases)
            for (const region of ['plane', 'backdrop']) {
                const r = row[region];
                if (!r) continue;
                const motionPixels = Math.hypot(r.expected[0] * 0.5 * size[0], r.expected[1] * 0.5 * size[1]);
                console.log(
                    `  ${row.name.padEnd(24)} ${region.padEnd(9)} ${fmt(motionPixels, 2)} ${fmt(r.ours.mean)} ${fmt(r.ours.max)} ${fmt(r.pixi.mean)} ${fmt(r.pixi.max)}   (${r.ours.pixels} px)`,
                );
                // pixi-3d's camera history is per render (docs/PIXI.md): reported, not a bar.
                if (KNOWN_LIMITS.has(row.name)) continue;
                if (!(r.ours.pixels > 0)) failures.push(`${row.name} / ${region}: no pixels measured`);
                // rgba16float stores the value: allow one f16 step of it (the GPU's conversion
                // needn't round to nearest; pixi-3d's own exact camera velocity shows the same step).
                const bar = EXACT + Math.hypot(f16Step(r.expected[0]) * 0.5 * size[0], f16Step(r.expected[1]) * 0.5 * size[1]);
                if (!(r.ours.mean <= bar)) failures.push(`${row.name} / ${region}: mean ${r.ours.mean.toFixed(4)} px (bar ${bar.toFixed(4)})`);
            }
        const records = logs(client);
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
        console.error(`\npixi check: ${failures.length} failure(s)\n  ${failures.join('\n  ')}`);
        process.exitCode = 1;
    } else console.log('\npixi check: MotionVectors is exact on every case');
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
