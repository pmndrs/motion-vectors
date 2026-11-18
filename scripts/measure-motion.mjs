#!/usr/bin/env node
/**
 * Motion lab meter (EXPERIMENTAL): the package's acceptance harness
 * (docs/PLAN.md §4). Drives lab/index.html over CDP;
 * every number is exact GPU readback.
 *
 * Experiments (--experiments, comma-separated):
 * - oracle: validates the oracle where three is known to be exact (courtyard
 *   track, camera segments): three and the oracle must agree to f16 noise.
 * - deformers: per object, endpoint error of zero motion / three / camera
 *   motion from depth against the oracle (static and animated positionNode
 *   displacement, morph, rigid control, frustum re-entry).
 * - cases: moves then holds (positionNode A→B, morph ramp, rigid control), per segment.
 * - ticks: the track with 1 / 2 / 4 renders per animation-loop tick.
 * - jitter: jitter switched on after warm-up, with one stable unjittered
 *   matrix vs the build-time trap (setProjectionMatrix(null) before).
 * - water: geometric swell (oracle) vs shading motion (glint, ripples: warp audit).
 * - track: per segment, the new bar (Q19's motion types in the lab's scene).
 * - cuts: the two cut detectors side by side (luma histogram EMD vs the
 *   depth-consistency share) on cuts and on the events that must not fire,
 *   with before / at / depth-map pictures per event, plus the GPU detector
 *   (`CutDetector`): its counts must equal the audit's pixel for pixel, its
 *   verdict must fire on every cut but the sky shot and on nothing else, and
 *   its cost (and motion from depth's) is timed where timestamps exist.
 * - probe: one --request (or an array), prints per-object errors.
 *
 * Usage:
 *   node scripts/measure-motion.mjs [--experiments …] [--label motion]
 *     [--width 1280] [--height 720] [--url http://127.0.0.1:5611] [--port 9411]
 *     [--adapter swiftshader] [--check]
 *
 * --check fails (exit 1) unless the conformance bars hold (CONTRACT.md):
 * three's velocity, the oracle and motion from depth agree on rigid and camera
 * motion, and the lab's controls read zero. CI runs it on SwiftShader.
 *
 * Writes results/<label>/<experiment>.json + PNGs (gitignored).
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

const PACKAGE = resolve(import.meta.dirname, '..');
const ALL = ['oracle', 'deformers', 'cases', 'ticks', 'jitter', 'water', 'track', 'cuts'];
const USAGE = `Usage: node scripts/measure-motion.mjs [options]
  --experiments <list>   ${ALL.join(',')},probe (default: all but probe)
  --width <px> --height <px>   render size (default 1280x720)
  --label <name>         output folder (default motion)
  --url <origin>         lab origin (default ${DEFAULT_BENCH_URL}); started if nothing answers
  --port <n>             Chrome DevTools port (default 9411)
  --request <json>       with probe: one lab request or an array
  --name <file>          with probe: screenshot name (default probe)
  --adapter swiftshader  software WebGPU (CI, no GPU)
  --check                exit 1 unless the conformance bars hold
  --reps <n>             with cost: interleaved repetitions (default 4)
Writes results/<label>/.`;
const cli = parseCliOrExit(process.argv.slice(2), ['experiments', 'width', 'height', 'label', 'url', 'port', 'request', 'name', 'adapter', 'check', 'reps'], USAGE);
const experiments = (cli.experiments ?? ALL.join(',')).split(',').filter(Boolean);
const width = Number(cli.width ?? 1280);
const height = Number(cli.height ?? 720);
const server = resolveServerUrl(cli.url, DEFAULT_BENCH_URL);
const port = parsePort(cli.port, '--port') ?? 9411;
const outputDirectory = join(PACKAGE, 'results', cli.label ?? 'motion');

/** Oracle comparison candidates, in the lab's order. */
const COMPARE = ['zero', 'three', 'camera'];
/** Warp audit candidates, in the lab's order. */
const AUDIT = ['zero', 'three', 'camera', 'oracle'];

//* Conformance Bars
/** Mean endpoint error a conforming source stays under where it is exact (CONTRACT.md). */
const EXACT_PIXELS = 0.01;
const failures = [];
/** Worst `WarpingError` disagreement with the audit over every run: scored/total pixels, `over` share, mean error. */
const warpParityMax = { exact: 0, over: 0, meanError: 0 };
/**
 * Records a bar for --check: `summary`'s candidate error must stay under EXACT_PIXELS.
 * @param {string} label - What is checked
 * @param {ReturnType<typeof compareSummary>} summary - An object's comparison
 * @param {number} candidate - 1 three, 2 camera motion from depth
 */
function expectExact(label, summary, candidate) {
    const error = summary?.candidates[candidate].error;
    if (error === undefined || error === null || !(error <= EXACT_PIXELS))
        failures.push(`${label}: ${COMPARE[candidate]} error ${error ?? 'n/a'} px (bar ${EXACT_PIXELS} px)`);
}

/** Mean relative `.w` error a conforming source stays under: one f16 step (2^-11) of rounding, twice. */
const DEPTH_EXACT = 2 ** -10;
/**
 * Records the contract's `.z`/`.w` bars for --check: no covered pixel with a
 * flag that contradicts the oracle, and `.w` within DEPTH_EXACT of the oracle's
 * previous depth where the candidate claims a previous position.
 * @param {string} label - What is checked
 * @param {ReturnType<typeof compareSummary>} summary - An object's comparison
 * @param {number} candidate - 1 the `velocity` slot, 2 camera motion from depth
 * @param {{ flags?: boolean }} [options] - `flags: false` skips the flag bar (honest no-previous flags the oracle doesn't share)
 */
function expectContract(label, summary, candidate, { flags = true } = {}) {
    const c = summary?.candidates[candidate];
    if (!c) return failures.push(`${label}: ${COMPARE[candidate]} not measured`);
    if (flags && c.wrongFlags !== 0) failures.push(`${label}: ${COMPARE[candidate]} ${c.wrongFlags} pixel(s) with a wrong .z flag`);
    if (c.depthError === null || !(c.depthError <= DEPTH_EXACT)) failures.push(`${label}: ${COMPARE[candidate]} .w error ${c.depthError ?? 'n/a'} (bar ${DEPTH_EXACT})`);
}

//* Summaries
const fmt = (value, digits = 3, pad = 8) => (value === null || value === undefined || !Number.isFinite(value) ? '—' : value.toFixed(digits)).padStart(pad);
const pct = (value, digits = 1, pad = 7) => fmt(value === null ? null : 100 * value, digits, pad);
const isMotion = (segment) => !/^(still|hold)/.test(segment);

/**
 * Pixel-weighted oracle comparison of one object over frames.
 * @returns {{ valid: number, motion: number|null, candidates: { name: string, error: number|null, over025: number|null, over1: number|null }[] } | null}
 */
function compareSummary(frames, object) {
    const covered = frames.map((frame) => frame.compare[object]).filter(Boolean);
    const rows = covered.filter((row) => row.valid > 0);
    const pixels = rows.reduce((total, row) => total + row.valid, 0);
    if (!pixels) return null;
    const weighted = (pick) => rows.reduce((total, row) => total + pick(row) * row.valid, 0) / pixels;
    const sum = (pick) => covered.reduce((total, row) => total + pick(row), 0);
    return {
        valid: pixels / frames.length,
        motion: weighted((row) => row.meanMotion),
        candidates: COMPARE.map((name, k) => {
            const depthPixels = sum((row) => row.candidates[k].depthPixels);
            return {
                name,
                error: weighted((row) => row.candidates[k].meanError),
                over025: weighted((row) => row.candidates[k].over025),
                over1: weighted((row) => row.candidates[k].over1),
                // .z/.w (CONTRACT.md): flags over every covered pixel, the first frame's too.
                wrongFlags: sum((row) => row.candidates[k].wrongFlags),
                depthError: depthPixels ? sum((row) => row.candidates[k].depthError * row.candidates[k].depthPixels) / depthPixels : null,
            };
        }),
    };
}

/** Frame-averaged warp audit per candidate. */
function auditSummary(frames) {
    const rows = frames.map((frame) => frame.audit).filter(Boolean);
    if (!rows.length) return null;
    const average = (pick) => rows.reduce((total, row) => total + pick(row), 0) / rows.length;
    return AUDIT.map((name, k) => ({
        name,
        residual: average((row) => row.candidates[k].meanResidual),
        unexplained: average((row) => row.candidates[k].unexplained),
        beaten: average((row) => row.candidates[k].beaten),
    }));
}

const COMPARE_HEADER = '  object/segment          px moved   zero err   three err  >0.25px  >1px   camera err  >0.25px  .z wrong   .w err';
function printCompare(label, summary) {
    if (!summary) return console.log(`  ${label.padEnd(22)}  (not visible)`);
    const [zero, three, camera] = summary.candidates;
    console.log(
        `  ${label.padEnd(22)} ${fmt(summary.motion)}  ${fmt(zero.error)}   ${fmt(three.error)}  ${pct(three.over025)} ${pct(three.over1, 1, 5)}  ${fmt(camera.error)}  ${pct(camera.over025)} ${String(camera.wrongFlags).padStart(9)} ${fmt(camera.depthError, 5)}`,
    );
}
function printAudit(label, rows) {
    if (!rows) return;
    console.log(`  ${label}: warp residual / unexplained%  ${rows.map((row) => `${row.name} ${fmt(row.residual, 4, 0)} / ${pct(row.unexplained, 2, 0)}`).join('   ')}`);
}

function bySegment(frames, segments) {
    return segments.map((segment) => ({ ...segment, frames: frames.filter((frame) => frame.frame >= segment.start && frame.frame <= segment.end) }));
}

//* Experiments
async function oracleCheck(run, screenshot) {
    const result = await run({ scene: 'track' });
    const camera = result.frames.filter((frame) => isMotion(frame.segment) && !frame.segment.startsWith('scene'));
    const objectMotion = result.frames.filter((frame) => frame.segment.startsWith('scene'));
    console.log(COMPARE_HEADER);
    const summary = {
        cameraStatic: compareSummary(camera, 'static'),
        cameraProps: compareSummary(camera, 'props'),
        objectProps: compareSummary(objectMotion, 'props'),
    };
    printCompare('camera motion / static', summary.cameraStatic);
    printCompare('camera motion / props', summary.cameraProps);
    printCompare('props moving / props', summary.objectProps);
    // Where three is exact, three and the oracle must agree: that validates the oracle.
    expectExact('oracle: camera motion, static', summary.cameraStatic, 1);
    expectExact('oracle: camera motion, static', summary.cameraStatic, 2);
    expectExact('oracle: camera motion, props', summary.cameraProps, 1);
    expectExact('oracle: moving props', summary.objectProps, 1);
    // Motion from depth fills the contract's .z/.w: camera-only and exact previous depth on static geometry.
    expectContract('oracle: camera motion, static', summary.cameraStatic, 2);
    expectContract('oracle: camera motion, props', summary.cameraProps, 2);
    await run({ scene: 'track', end: 60, view: 'audit-motion', heatCandidate: 3 });
    await screenshot('oracle-track-f60');
    return summary;
}

async function deformers(run, screenshot) {
    const result = await run({ scene: 'deformers' });
    console.log(COMPARE_HEADER);
    const summary = {};
    for (const name of Object.values(result.objects)) {
        summary[name] = compareSummary(result.frames, name);
        printCompare(name, summary[name]);
    }
    // The re-entry frames: the ball is drawn again after leaving the frustum.
    const reentry = result.frames
        .filter((frame) => frame.compare.reentry?.valid > 0)
        .map((frame) => ({ frame: frame.frame, error: frame.compare.reentry.candidates[1].meanError }))
        .sort((a, b) => b.error - a.error)
        .slice(0, 3);
    console.log(`  reentry: worst three frames ${reentry.map((row) => `${row.frame} (${fmt(row.error, 2, 0)} px)`).join(', ')}`);
    expectExact('deformers: rigid box', summary.rigid, 1);
    summary.audit = auditSummary(result.frames);
    printAudit('whole frame', summary.audit);
    for (const [view, candidate] of [['output', 1], ['audit-motion', 1], ['audit-motion', 3]]) {
        await run({ scene: 'deformers', end: 140, view, heatCandidate: candidate });
        await screenshot(`deformers-${view}${view === 'output' ? '' : `-${AUDIT[candidate]}`}-f140`);
    }
    return { summary, reentry };
}

async function cases(run) {
    const result = await run({ scene: 'cases' });
    const summary = {};
    console.log(COMPARE_HEADER);
    for (const name of Object.values(result.objects).filter((object) => object !== 'static'))
        for (const segment of bySegment(result.frames, result.segments)) {
            const key = `${name} ${segment.name}`;
            summary[key] = compareSummary(segment.frames, name);
            printCompare(key, summary[key]);
            if (name === 'rigid-move') expectExact(`cases: ${key}`, summary[key], 1);
        }
    return summary;
}

async function ticks(run) {
    const summary = {};
    console.log(COMPARE_HEADER);
    for (const rendersPerTick of [1, 2, 4]) {
        const result = await run({ scene: 'track', rendersPerTick });
        const camera = result.frames.filter((frame) => isMotion(frame.segment) && !frame.segment.startsWith('scene'));
        summary[rendersPerTick] = compareSummary(camera, 'static');
        printCompare(`${rendersPerTick} render(s)/tick`, summary[rendersPerTick]);
    }
    return summary;
}

async function jitter(run) {
    // Still camera: the correct motion is zero everywhere once jitter is on.
    const summary = {};
    console.log(COMPARE_HEADER);
    for (const velocityProjection of ['stable', 'trap']) {
        const result = await run({ scene: 'track', end: 29, jitter: 10, velocityProjection, fresh: true });
        const after = result.frames.filter((frame) => frame.jittered);
        summary[velocityProjection] = compareSummary(after, 'static');
        printCompare(`jitter on, ${velocityProjection}`, summary[velocityProjection]);
    }
    expectExact('jitter: stable matrix, still scene', summary.stable, 1);
    return summary;
}

/**
 * This package's three node (src/three) against three's `velocity`, on every
 * oracle case. The node fills the lab's `velocity` slot (`motion: 'node'`),
 * so both read as candidate 1 of their own runs. Phase 2's exit: every object
 * and segment within the exact bar.
 */
async function node(run) {
    const summary = {};
    // A --check run (CI, SwiftShader) only needs the node's bars: it skips
    // three's runs and the un-driven rows, and shortens the track to its two
    // dolly segments (several renders per tick still change history there).
    const quick = cli.check === true;
    const threeRun = (request) => (quick ? Promise.resolve(null) : run(request));
    const header = '  scene/object/segment              px moved   three err   node err  >0.25px  >1px  .z wrong   .w err';
    const row = (label, three, ours) => {
        const c = ours?.candidates[1];
        summary[label] = { three: three?.candidates[1].error ?? null, node: c?.error ?? null, motion: ours?.motion ?? null, wrongFlags: c?.wrongFlags ?? null, depthError: c?.depthError ?? null };
        if (!ours) return console.log(`  ${label.padEnd(32)}  (not visible)`);
        console.log(
            `  ${label.padEnd(32)} ${fmt(ours.motion)}  ${fmt(three?.candidates[1].error)}   ${fmt(c.error)}  ${pct(c.over025)} ${pct(c.over1, 1, 5)} ${String(c.wrongFlags).padStart(9)} ${fmt(c.depthError, 5)}`,
        );
        expectExact(`node: ${label}`, ours, 1);
        // The re-entering ball is honestly flagged "no previous" on its first frame back; the oracle knows where it was.
        expectContract(`node: ${label}`, ours, 1, { flags: label !== 'deformers reentry' });
    };
    console.log(header);
    for (const scene of ['deformers', 'cases', 'water']) {
        const three = await threeRun({ scene });
        const ours = await run({ scene, motion: 'node' });
        for (const name of Object.values(ours.objects)) {
            if (scene === 'cases' && name !== 'static') {
                const threeSegments = three ? bySegment(three.frames, three.segments) : null;
                bySegment(ours.frames, ours.segments).forEach((segment, i) =>
                    row(`${scene} ${name} ${segment.name}`, threeSegments && compareSummary(threeSegments[i].frames, name), compareSummary(segment.frames, name)),
                );
            } else row(`${scene} ${name}`, three && compareSummary(three.frames, name), compareSummary(ours.frames, name));
        }
    }
    const trackEnd = quick ? 175 : undefined;
    for (const rendersPerTick of quick ? [2] : [1, 2, 4]) {
        const pick = (result) => result && compareSummary(result.frames.filter((frame) => isMotion(frame.segment) && !frame.segment.startsWith('scene')), 'static');
        const three = await threeRun({ scene: 'track', rendersPerTick, end: trackEnd });
        const ours = await run({ scene: 'track', rendersPerTick, motion: 'node', end: trackEnd });
        row(`track ${rendersPerTick} render(s)/tick`, pick(three), pick(ours));
        if (quick) continue;
        // Not driven: the node on three's frame counter, as an app that never calls beginFrame().
        const undriven = await run({ scene: 'track', rendersPerTick, motion: 'node', driven: false });
        const label = `track ${rendersPerTick}/tick, no beginFrame`;
        if (rendersPerTick === 1) row(label, pick(three), pick(undriven));
        else {
            const summaryRow = pick(undriven);
            summary[label] = { three: pick(three)?.candidates[1].error ?? null, node: summaryRow?.candidates[1].error ?? null };
            console.log(`  ${label.padEnd(32)} ${fmt(summaryRow?.motion)}  ${fmt(pick(three)?.candidates[1].error)}   ${fmt(summaryRow?.candidates[1].error)}  (expected: needs beginFrame)`);
        }
    }
    {
        const pick = (result) => result && compareSummary(result.frames.filter((frame) => frame.jittered), 'static');
        const three = await threeRun({ scene: 'track', end: 29, jitter: 10, velocityProjection: 'trap', fresh: true });
        const ours = await run({ scene: 'track', end: 29, jitter: 10, motion: 'node', fresh: true });
        row('track jitter on (three: trap)', pick(three), pick(ours));
    }
    return summary;
}

/**
 * R11: the node's cost on the scene render against three's `velocity`, as GPU
 * time of the whole scene render (shadow + MRT passes, three's timestamps),
 * interleaved three/node/node/three per repetition so clock drift cancels.
 */
async function cost(run) {
    const summary = {};
    const reps = Number(cli.reps ?? 4);
    const warmup = 10;
    const mean = (result) => {
        const values = result.frames.filter((frame) => frame.frame >= warmup && Number.isFinite(frame.renderMs)).map((frame) => frame.renderMs);
        return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
    };
    console.log('  scene        three ms    node ms  median Δus    Δ %   (per repetition: node − three, µs; % of three\'s scene render)');
    for (const scene of ['deformers', 'water', 'track']) {
        const end = scene === 'track' ? 240 : undefined;
        const pairs = [];
        for (let rep = 0; rep < reps; rep++) {
            const order = rep % 2 === 0 ? ['three', 'node'] : ['node', 'three'];
            const times = {};
            for (const motion of [...order, ...[...order].reverse()]) {
                const result = await run({ scene, end, motion, timing: true });
                (times[motion] ??= []).push(mean(result));
            }
            const avg = (list) => list.reduce((a, b) => a + b, 0) / list.length;
            pairs.push({ three: avg(times.three), node: avg(times.node) });
        }
        const three = pairs.reduce((a, p) => a + p.three, 0) / pairs.length;
        const ours = pairs.reduce((a, p) => a + p.node, 0) / pairs.length;
        const deltas = pairs.map((p) => (p.node - p.three) * 1000);
        const median = [...deltas].sort((a, b) => a - b)[Math.floor(deltas.length / 2)];
        summary[scene] = { threeMs: three, nodeMs: ours, deltaUs: deltas, medianDeltaUs: median, medianDeltaPct: median / (10 * three) };
        console.log(
            `  ${scene.padEnd(10)} ${fmt(three, 3, 9)}  ${fmt(ours, 3, 9)}  ${fmt(median, 1, 9)}  ${fmt(summary[scene].medianDeltaPct, 2, 7)}   (${deltas.map((d) => d.toFixed(0)).join(', ')})`,
        );
    }
    return summary;
}

async function water(run, screenshot) {
    const result = await run({ scene: 'water' });
    console.log(COMPARE_HEADER);
    const summary = {};
    for (const name of Object.values(result.objects)) {
        summary[name] = compareSummary(result.frames, name);
        printCompare(name, summary[name]);
    }
    summary.audit = auditSummary(result.frames);
    printAudit('whole frame', summary.audit);
    for (const [view, candidate] of [['output', 1], ['audit', 3], ['audit-heat', 3]]) {
        await run({ scene: 'water', end: 120, view, heatCandidate: candidate });
        await screenshot(`water-${view}-f120`);
    }
    return summary;
}

async function track(run, screenshot) {
    const result = await run({ scene: 'track' });
    const rows = bySegment(result.frames, result.segments).map((segment) => ({
        segment: segment.name,
        static: compareSummary(segment.frames, 'static'),
        props: compareSummary(segment.frames, 'props'),
        audit: auditSummary(segment.frames),
    }));
    console.log('  segment              static: moved  three  camera  zero | props: moved  three  camera | warp: three  camera  oracle  zero');
    for (const row of rows) {
        const c = (summary, k) => summary?.candidates[k].error ?? null;
        const a = (k) => row.audit?.[k].residual ?? null;
        console.log(
            `  ${row.segment.padEnd(20)} ${fmt(row.static?.motion, 2, 7)} ${fmt(c(row.static, 1), 3, 6)} ${fmt(c(row.static, 2), 3, 7)} ${fmt(c(row.static, 0), 2, 6)} |` +
                ` ${fmt(row.props?.motion, 2, 6)} ${fmt(c(row.props, 1), 3, 6)} ${fmt(c(row.props, 2), 3, 7)} | ${fmt(a(1), 4, 7)} ${fmt(a(2), 4, 7)} ${fmt(a(3), 4, 7)} ${fmt(a(0), 4, 7)}`,
        );
    }
    for (const frame of [60, 510]) {
        await run({ scene: 'track', end: frame, view: 'output' });
        await screenshot(`track-output-f${frame}`);
    }
    return { segments: rows, maxEMD: Math.max(...result.frames.map((frame) => frame.cutEMD ?? 0)), maxJS: Math.max(...result.frames.map((frame) => frame.cutJS ?? 0)) };
}

/** What each cut-timeline segment is, and whether a detector should fire on it. */
const CUT_EVENTS = {
    'cut-hard': { cut: true, label: 'Hard cut to another angle' },
    'cut-similar': { cut: true, label: 'Cut between similar shots' },
    fade: { cut: false, label: 'Fade to 5 % and back' },
    flash: { cut: false, label: 'One-frame ×6 flash' },
    'pan-fast': { cut: false, label: 'Pan, 3°/frame' },
    'cut-back': { cut: true, label: 'Cut back after a pan' },
    'whip-pan': { cut: false, label: 'Whip pan, 9°/frame' },
    'cut-back-2': { cut: true, label: 'Cut back after a whip pan' },
    sweep: { cut: false, label: 'Near occluder sweeps across' },
    'lights-off': { cut: false, label: 'Lights step to 15 %' },
    'lights-on': { cut: false, label: 'Lights step back up' },
    'cut-sky': { cut: true, label: 'Cut to a sky-heavy shot' },
    'cut-sky-2': { cut: true, label: 'Cut between two sky-heavy shots' },
    'cut-palette': { cut: true, label: 'Cut to another location (palette)' },
};
/** Thresholds the comparison is read at (EMD: Phase 0's, fit on Q10). */
const CUT_THRESHOLDS = { emd: 0.5, depth: 0.2 };
/** The one cut whose history stays valid (MOTION.md §7): the detector rightly keeps it. */
const CUT_KEEPS_HISTORY = new Set(['cut-sky']);

/**
 * The GPU cut counts (the lab's `MotionFromDepthPass`) against the audit's
 * (the analyzer's own motion from depth, same test), in pixels; `null` when a
 * frame lacks either.
 */
function cutParity(frame) {
    const depth = frame.cutDepth;
    const gpu = frame.cutGpu;
    if (!depth || !gpu) return null;
    // Background (no surface) is left out of both counts; shares × pixels are integers.
    const inconsistent = depth.noneValid - depth.backgroundUnscored;
    const surface = 1 - depth.background;
    return Math.max(
        Math.abs(Math.round(inconsistent * depth.pixels) - gpu.inconsistentPixels),
        Math.abs(Math.round(surface * depth.pixels) - gpu.surfacePixels),
    );
}

/**
 * The lab's `WarpingError` (camera motion from depth, masked) against the
 * audit's camera candidate (slot 1, candidate 2: the same test). Scored and
 * total pixels must match exactly. Two separately compiled shaders can still
 * disagree on a pixel sitting exactly at the threshold or at a fixed-point .5
 * (measured: `over` up to 4 px, the error sum a few units of ~10⁷), so those
 * get tolerances far below anything a bench reads: `over` within 1e-5 of the
 * pixels, the mean error within 1e-6. `null` when a frame lacks either.
 */
function warpParity(frame) {
    const audit = frame.cutDepth;
    const warp = frame.warp;
    if (!audit || !warp?.hasPrevious) return null;
    const camera = audit.candidates[2];
    const valid = Math.round(camera.valid * audit.pixels);
    return {
        exact: Math.max(Math.abs(valid - warp.counts.scored), Math.abs(audit.pixels - warp.counts.pixels)),
        over: Math.abs(Math.round(camera.unexplained * audit.pixels) - warp.counts.over) / audit.pixels,
        meanError: Math.abs(camera.meanResidual - warp.meanError),
    };
}

/** Pools warping results over frames: total error over total scored pixels, mean shares. */
function poolWarp(results) {
    const measured = results.filter((result) => result?.hasPrevious);
    if (!measured.length) return null;
    const sum = (key) => measured.reduce((total, result) => total + result.counts[key], 0);
    const pixels = sum('pixels');
    return { frames: measured.length, meanError: sum('errorSum') / 1024 / Math.max(sum('scored'), 1), scored: sum('scored') / pixels, over: sum('over') / pixels };
}

/**
 * Warping error as a bench metric: does it rank motion fields the way the
 * oracle does, with no ground truth? Unmasked, so every field is scored on
 * (nearly) the same pixels: camera motion from depth (static world only),
 * three's `velocity` and this package's node, on the scenes with moving and
 * deforming objects. Bars: on deformers and cases, the node (exact there,
 * MOTION.md §8) reads below both. Water is reported only (see below).
 */
async function warp(run) {
    const summary = {};
    console.log('  scene        frames   camera err  three err   node err   scored (camera / three / node)');
    for (const scene of ['deformers', 'cases', 'water']) {
        const threeRun = await run({ scene, warp: true });
        const nodeRun = await run({ scene, motion: 'node', warp: true });
        const camera = poolWarp(nodeRun.frames.map((frame) => frame.warpUnmasked?.camera));
        const three = poolWarp(threeRun.frames.map((frame) => frame.warpUnmasked?.engine));
        const node = poolWarp(nodeRun.frames.map((frame) => frame.warpUnmasked?.engine));
        summary[scene] = { camera, three, node };
        console.log(
            `  ${scene.padEnd(12)} ${String(node.frames).padStart(6)}  ${fmt(camera.meanError, 5, 10)} ${fmt(three.meanError, 5, 10)} ${fmt(node.meanError, 5, 10)}   ${pct(camera.scored)} / ${pct(three.scored)} / ${pct(node.scored)}`,
        );
        // Water is reported, not barred: its glints and ripples don't move with the surface (MOTION.md §7, "Water"),
        // so the exact geometric motion is no better at warping its image than camera motion (MOTION.md §9).
        if (scene === 'water') continue;
        if (node.meanError > camera.meanError) failures.push(`warp ${scene}: node ${node.meanError} above camera motion ${camera.meanError}`);
        if (node.meanError > three.meanError) failures.push(`warp ${scene}: node ${node.meanError} above three ${three.meanError}`);
    }
    return summary;
}

async function cuts(run, screenshot) {
    const result = await run({ scene: 'cuts' });
    const series = result.frames.map((frame) => {
        const depth = frame.cutDepth;
        const entering = depth?.entering ?? 0;
        const background = depth?.background ?? 0;
        // Background (no surface) carries no evidence either way: leave it out of
        // both counts. Content from off-screen stays in: at a cut, most of the new
        // shot was off-screen in the old camera. Under 5 % surface: no verdict.
        const surface = 1 - background;
        const inconsistent = depth ? depth.noneValid - depth.backgroundUnscored : 0;
        const gpu = frame.cutGpu;
        const parity = cutParity(frame);
        return {
            frame: frame.frame,
            segment: frame.segment,
            cut: frame.cut,
            js: frame.cutJS,
            emd: frame.cutEMD,
            depthRaw: depth?.noneValid ?? null,
            entering,
            background,
            depth: depth && surface >= 0.05 ? inconsistent / surface : null,
            gpu: gpu?.share ?? null,
            gpuCut: gpu?.cut ?? null,
            parity,
            cameraMotionMs: frame.cameraMotionMs ?? null,
            cutMs: frame.cutMs ?? null,
            warpMs: frame.warpMs ?? null,
        };
    });
    const max = (rows, key) => Math.max(0, ...rows.map((row) => row[key] ?? 0));
    const events = [];
    for (const segment of result.segments) {
        const info = CUT_EVENTS[segment.name];
        if (!info) continue;
        const rows = series.filter((row) => row.segment === segment.name);
        // A cut is judged on its cut frame; anything else on its worst frame.
        const judged = info.cut ? rows.filter((row) => row.cut) : rows;
        const peak = (key) => judged.reduce((best, row) => ((row[key] ?? -1) > (best?.[key] ?? -1) ? row : best), null);
        const at = peak('depth');
        const gpuCut = judged.some((row) => row.gpuCut === true);
        events.push({
            segment: segment.name,
            ...info,
            start: segment.start,
            emd: max(judged, 'emd'),
            js: max(judged, 'js'),
            depth: max(judged, 'depth'),
            depthRaw: max(judged, 'depthRaw'),
            entering: max(judged, 'entering'),
            background: max(judged, 'background'),
            emdFrame: peak('emd')?.frame,
            depthFrame: at?.frame,
            gpu: max(judged, 'gpu'),
            gpuCut,
        });
        // --check bars: a cut fires on its cut frame (but the sky shot, whose history stays valid); nothing else fires.
        const expected = info.cut && !CUT_KEEPS_HISTORY.has(segment.name);
        if (gpuCut !== expected) failures.push(`cut detector: ${info.label} ${gpuCut ? 'fired' : 'missed'} (GPU share ${max(judged, 'gpu').toFixed(3)})`);
    }
    // Holds after cuts and the plain holds: the steady-state floor of each signal.
    const quiet = series.filter((row) => !CUT_EVENTS[row.segment] || (CUT_EVENTS[row.segment].cut && !row.cut));
    const floor = { emd: max(quiet, 'emd'), depth: max(quiet, 'depth'), depthRaw: max(quiet, 'depthRaw'), gpu: max(quiet, 'gpu') };
    const quietFires = quiet.filter((row) => row.gpuCut === true).map((row) => row.frame);
    if (quietFires.length) failures.push(`cut detector: fired on quiet frame(s) ${quietFires.join(', ')}`);
    const parity = Math.max(0, ...series.map((row) => row.parity ?? 0));
    if (series.some((row) => row.parity === null)) failures.push('cut detector: a measured frame has no GPU verdict');
    if (parity !== 0) failures.push(`cut detector: counts differ from the audit's by up to ${parity} pixel(s)`);
    const verdict = (value, threshold, cut) => (value > threshold ? (cut ? 'hit' : 'FALSE') : cut ? 'MISS' : 'ok');
    const gpuVerdict = (row) => (row.gpuCut ? (row.cut ? 'hit' : 'FALSE') : row.cut ? (CUT_KEEPS_HISTORY.has(row.segment) ? 'keep' : 'MISS') : 'ok');
    console.log(
        `  ${'event'.padEnd(36)} ${'EMD'.padStart(8)}       ${'depth'.padStart(8)}       ${'GPU'.padStart(8)}       ${'raw'.padStart(8)} ${'enter'.padStart(8)} ${'bg'.padStart(8)}`,
    );
    for (const row of events)
        console.log(
            `  ${row.label.padEnd(36)} ${fmt(row.emd)} ${verdict(row.emd, CUT_THRESHOLDS.emd, row.cut).padEnd(5)} ${fmt(row.depth)} ${verdict(row.depth, CUT_THRESHOLDS.depth, row.cut).padEnd(5)} ${fmt(row.gpu)} ${gpuVerdict(row).padEnd(5)} ${fmt(row.depthRaw)} ${fmt(row.entering)} ${fmt(row.background)}`,
        );
    console.log(`  ${'quiet frames (max)'.padEnd(36)} ${fmt(floor.emd)}       ${fmt(floor.depth)}       ${fmt(floor.gpu)}       ${fmt(floor.depthRaw)}`);
    console.log(`  GPU detector vs audit: counts differ by ${parity} pixel(s) at most, over ${series.length} frames`);
    const timed = (key) => series.map((row) => row[key]).filter((value) => value !== null);
    const cost = {};
    for (const [key, label] of [
        ['cameraMotionMs', 'motion from depth'],
        ['cutMs', 'cut detector'],
        ['warpMs', 'warping error'],
    ]) {
        const values = timed(key).sort((a, b) => a - b);
        if (!values.length) continue;
        cost[key] = { median: values[values.length >> 1], mean: values.reduce((a, b) => a + b, 0) / values.length, max: values[values.length - 1], frames: values.length };
        console.log(`  ${label.padEnd(18)} GPU µs: median ${(1000 * cost[key].median).toFixed(1)}, mean ${(1000 * cost[key].mean).toFixed(1)}, max ${(1000 * cost[key].max).toFixed(1)} (${values.length} frames)`);
    }
    if (cli.check) return { thresholds: CUT_THRESHOLDS, events, floor, parity, cost, series };

    // Pictures per event: the frame before, the event frame, and where depth says "no motion reprojects".
    const shots = [];
    for (const row of events) {
        const frame = row.cut ? row.start : (row.depthFrame ?? row.emdFrame ?? row.start);
        for (const [view, end, name] of [
            ['output', frame - 1, 'before'],
            ['output', frame, 'at'],
            ['audit-cut', frame, 'depth'],
        ]) {
            await run({ scene: 'cuts', end, start: end, view });
            await screenshot(`cuts-${row.segment}-${name}`);
        }
        shots.push({ segment: row.segment, frame });
    }
    return { thresholds: CUT_THRESHOLDS, events, floor, parity, cost, series, shots };
}

async function probe(run, screenshot) {
    const parsed = JSON.parse(cli.request ?? '{"scene":"deformers"}');
    const results = [];
    for (const request of Array.isArray(parsed) ? parsed : [parsed]) {
        const result = await run(request);
        console.log(`  ${JSON.stringify(request)}`);
        console.log(COMPARE_HEADER);
        for (const name of Object.values(result.objects)) printCompare(name, compareSummary(result.frames, name));
        printAudit('whole frame', auditSummary(result.frames));
        results.push(result);
    }
    await screenshot(cli.name ?? 'probe');
    return Array.isArray(parsed) ? results : results[0];
}

const EXPERIMENTS = { oracle: oracleCheck, deformers, cases, ticks, jitter, water, track, cuts, node, warp, cost, probe };

//* Harness
async function main() {
    await mkdir(outputDirectory, { recursive: true });
    let viteServer = null;
    let chrome = null;
    let profile = null;
    let client = null;
    try {
        try {
            await waitForUrl(server.origin, { attempts: 1 });
        } catch {
            viteServer = spawnVite('lab/vite.config.ts', server);
            await waitForUrl(server.origin, { child: viteServer });
        }
        profile = await mkdtemp(join(tmpdir(), 'motion-lab-'));
        chrome = spawn(
            browserExecutable(),
            [
                '--headless=new',
                '--enable-unsafe-webgpu',
                '--disable-background-timer-throttling',
                '--disable-renderer-backgrounding',
                `--remote-debugging-port=${port}`,
                `--user-data-dir=${profile}`,
                `--window-size=${width + 40},${height + 40}`,
                '--force-device-scale-factor=1',
                ...(cli.adapter === 'swiftshader' ? ['--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan'] : []),
                ...(process.env.CI ? ['--no-sandbox'] : []),
                // Extra Chrome flags (space-separated), e.g. the compositor backend headless Linux needs.
                ...(process.env.CHROME_FLAGS ?? '').split(' ').filter(Boolean),
                'about:blank',
            ],
            // Chrome's own stderr (GPU process crashes, Dawn errors) goes to a file next to the results.
            { stdio: ['ignore', 'ignore', openSync(join(outputDirectory, 'chrome.log'), 'w')] },
        );
        const cdpBase = `http://127.0.0.1:${port}`;
        await waitForUrl(`${cdpBase}/json/version`);
        const created = await fetch(`${cdpBase}/json/new?about:blank`, { method: 'PUT' }).then((response) => response.json());
        client = await CDP.connect(created.webSocketDebuggerUrl);
        await Promise.all([client.send('Page.enable'), client.send('Runtime.enable'), client.send('Log.enable')]);
        // Pin the viewport to the canvas, or headless Chrome clips its bottom rows.
        await client.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
        // Screenshots need the canvas; a --check run only reads GPU buffers, so it skips the present (lab main.ts).
        await client.send('Page.navigate', { url: `${server.origin}/index.html?present=${cli.check ? 0 : 1}&timing=${experiments.includes('cost') ? 1 : 0}` });
        for (let attempt = 0; ; attempt++) {
            if ((await client.evaluate('window.__motionLab?.ready === true')) === true) break;
            if (attempt > 600) throw new Error(`Timed out waiting for the motion lab.\n${logs(client).join('\n')}`);
            await new Promise((resolveWait) => setTimeout(resolveWait, 100));
        }

        const run = async (request) => {
            const result = await client.send(
                'Runtime.evaluate',
                { expression: `window.__motionLab.run(${JSON.stringify({ width, height, ...request })})`, awaitPromise: true, returnByValue: true },
                30 * 60 * 1000,
            );
            if (result.exceptionDetails)
                throw new Error(`${result.exceptionDetails.exception?.description ?? result.exceptionDetails.text}\n${logs(client).slice(-40).join('\n')}`);
            // Every run checks the three helper (motion from depth from three's camera,
            // jitter included) against the audit, pixel for pixel.
            const value = result.result.value;
            const parity = Math.max(0, ...value.frames.map((frame) => cutParity(frame) ?? 0));
            if (parity !== 0) failures.push(`MotionFromDepthPass (${request.scene}): cut counts differ from the audit's by up to ${parity} pixel(s)`);
            const warpRows = value.frames.map(warpParity).filter(Boolean);
            const most = (key) => Math.max(0, ...warpRows.map((row) => row[key]));
            for (const key of Object.keys(warpParityMax)) warpParityMax[key] = Math.max(warpParityMax[key], most(key));
            if (most('exact') !== 0) failures.push(`WarpingError (${request.scene}): scored pixels differ from the audit's camera candidate by up to ${most('exact')}`);
            if (most('over') > 1e-5) failures.push(`WarpingError (${request.scene}): 'over' differs by ${most('over').toExponential(2)} of the pixels`);
            if (most('meanError') > 1e-6) failures.push(`WarpingError (${request.scene}): mean error differs by ${most('meanError').toExponential(2)}`);
            return value;
        };
        const screenshot = async (name) => {
            if (cli.check) return; // nothing is presented in a check run
            const shot = await client.send('Page.captureScreenshot', { format: 'png', clip: { x: 0, y: 0, width, height, scale: 1 } });
            await writeFile(join(outputDirectory, `${name}.png`), Buffer.from(shot.data, 'base64'));
        };

        for (const experiment of experiments) {
            const measure = EXPERIMENTS[experiment];
            if (!measure) throw new Error(`Unknown experiment ${experiment}`);
            const started = Date.now();
            console.log(`\n== ${experiment}`);
            const record = await measure(run, screenshot);
            await writeFile(join(outputDirectory, `${experiment}.json`), `${JSON.stringify(record, null, 2)}\n`);
            console.log(`  (${((Date.now() - started) / 1000).toFixed(0)} s)`);
        }
        const records = logs(client);
        await writeFile(join(outputDirectory, 'browser-log.txt'), `${records.join('\n')}\n`);
        if (records.some((line) => /error|exception|validation|invalid/i.test(line))) console.warn(`\nbrowser log:\n${records.join('\n')}`);
        console.log(`\nartifacts: ${outputDirectory}`);
        console.log(
            `\nWarpingError vs the audit's camera candidate, over every run: scored ${warpParityMax.exact} px apart, 'over' ${warpParityMax.over.toExponential(2)} of the pixels, mean error ${warpParityMax.meanError.toExponential(2)}`,
        );
        if (cli.check) {
            if (failures.length) {
                console.error(`\nconformance: ${failures.length} bar(s) failed\n  ${failures.join('\n  ')}`);
                process.exitCode = 1;
            } else console.log('\nconformance: every bar holds');
        }
    } finally {
        client?.close();
        await stopChild(chrome);
        await removeTempDirectory(profile, 'Chrome profile');
        await stopChild(viteServer);
    }
}

function logs(client) {
    return client.events
        .filter((event) => event.method === 'Log.entryAdded' || event.method === 'Runtime.exceptionThrown' || event.method === 'Runtime.consoleAPICalled')
        .map((event) => {
            if (event.method === 'Log.entryAdded') return `[${event.params.entry.level}] ${event.params.entry.text}`;
            if (event.method === 'Runtime.exceptionThrown')
                return `[exception] ${event.params.exceptionDetails.exception?.description ?? event.params.exceptionDetails.text}`;
            return `[console.${event.params.type}] ${event.params.args.map((arg) => arg.value ?? arg.description).join(' ')}`;
        });
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
