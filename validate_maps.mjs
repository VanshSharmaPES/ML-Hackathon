// Syntax and content verification only: this does not claim browser rendering.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
const directory = path.join(import.meta.dirname, 'outputs', 'maps');
const results = [];
for (const filename of fs.readdirSync(directory).filter(name => name.endsWith('.html'))) {
  const html = fs.readFileSync(path.join(directory, filename), 'utf8');
  const scripts = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)]
    .map(match => match[1]).filter(text => text.trim());
  if (!scripts.length) throw Error(`No inline map script in ${filename}`);
  for (const script of scripts) new vm.Script(script, { filename });
  for (const required of ['L.map(', 'L.geoJson(', 'L.control.layers(', 'Raw GPS observations',
    'Road network', 'Reference route', 'Large time gaps', 'PathSnap development map']) {
    if (!html.includes(required)) throw Error(`${filename} missing ${required}`);
  }
  results.push({ filename, bytes: Buffer.byteLength(html), inlineScripts: scripts.length, status: 'passed' });
}
if (results.length < 4) throw Error('Expected at least four representative maps');
fs.writeFileSync(path.join(import.meta.dirname, 'outputs', 'map_validation.json'), JSON.stringify(results, null, 2));
console.log(JSON.stringify(results, null, 2));
