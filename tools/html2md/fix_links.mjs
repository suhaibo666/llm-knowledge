// Rewrite Obsidian wiki-links that point at the now-deleted .html pages: [[x.html]] -> [[x]],
// [[x.html|alias]] -> [[x|alias]], [[x.html\|alias]] -> [[x\|alias]].
// ONLY touches [[ ... ]] links — external markdown URLs (.html) and backticked paths are left alone.
// changelog.md is excluded (its prose quotes the literal "[[*.html]]" pattern).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const WIKI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'wiki');
const LINK_RE = /\[\[([^\]]*?)\.html((?:\\?\|[^\]]*)?)\]\]/g;

function walk(dir, out = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, out);
    else if (e.isFile() && e.name.endsWith('.md')) out.push(p);
  }
  return out;
}

let total = 0;
for (const f of walk(WIKI)) {
  if (path.basename(f) === 'changelog.md') continue;
  const src = fs.readFileSync(f, 'utf8');
  let n = 0;
  const out = src.replace(LINK_RE, (_m, a, b) => { n++; return `[[${a}${b}]]`; });
  if (n > 0) {
    fs.writeFileSync(f, out, 'utf8');
    total += n;
    console.log(`${n}\t${path.relative(WIKI, f)}`);
  }
}
console.log(`\nTotal wiki-link rewrites: ${total}`);
