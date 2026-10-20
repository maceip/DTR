import {build} from 'esbuild';
import {cp, readFile, rm} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {litertLmBindings} from './patches/litert-lm-0.18.1.mjs';

const root = path.dirname(fileURLToPath(import.meta.url));
process.chdir(root);
for (const [name, expected] of [['@litertjs/core', '2.5.3'], ['@litert-lm/core', '0.18.1']]) {
  const metadata = JSON.parse(await readFile(path.join('node_modules', name, 'package.json')));
  if (metadata.version !== expected) throw new Error(`Update capability contracts before changing ${name} ${metadata.version}`);
}
await rm('dist', {recursive: true, force: true});
execFileSync(process.execPath, ['node_modules/typescript/bin/tsc', '--emitDeclarationOnly'], {stdio: 'inherit'});
await build({entryPoints: ['src/index.ts'], outdir: 'dist', bundle: true,
  format: 'esm', splitting: true, platform: 'browser', target: ['es2022'], sourcemap: true, logLevel: 'info'});
await build({entryPoints: ['src/browser-worker.ts'], outfile: 'dist/browser-worker.js', bundle: true,
  format: 'iife', platform: 'browser', target: ['es2022'], sourcemap: true, logLevel: 'info', plugins: [litertLmBindings]});
for (const [name, directory] of [['@litertjs/core', 'litert'], ['@litert-lm/core', 'litert-lm']]) {
  await cp(path.join('node_modules', name, 'wasm'), path.join('dist/wasm', directory), {recursive: true});
}
await cp('licenses', 'dist/licenses', {recursive: true});
await cp('THIRD_PARTY_NOTICES.md', 'dist/THIRD_PARTY_NOTICES.md');
console.log('Built the SDK, isolated inference worker, and local WASM assets.');
