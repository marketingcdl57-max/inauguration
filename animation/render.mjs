// Rend index.html en vidéo MP4 (1080x1350, 30 i/s) image par image.
// Usage : node render.mjs [durée_s] [sortie.mp4]
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const dir = path.dirname(fileURLToPath(import.meta.url));
const DUR = Number(process.argv[2] ?? 11);
const OUT = process.argv[3] ?? path.join(dir, 'inauguration-lavenir.mp4');
const FPS = 30, W = 1080, H = 1350;
const ONLY = process.env.FRAMES?.split(',').map(Number);   // ex. FRAMES=1,3.5 → PNG de contrôle

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined });
const page = await browser.newPage({ viewport: { width: W, height: H } });
await page.goto(pathToFileURL(path.join(dir, 'index.html')).href);
await page.evaluate(() => document.fonts.ready);

if (ONLY) {
  for (const t of ONLY) {
    await page.evaluate((ms) => window.seek(ms), t * 1000);
    await page.screenshot({ path: path.join(dir, `frame-${t}.png`) });
  }
} else {
  const ff = spawn('ffmpeg', ['-loglevel', 'error', '-y', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT],
    { stdio: ['pipe', 'inherit', 'inherit'] });
  const n = Math.round(DUR * FPS);
  for (let i = 0; i < n; i++) {
    await page.evaluate((ms) => window.seek(ms), (i * 1000) / FPS);
    ff.stdin.write(await page.screenshot({ type: 'png' }));
  }
  ff.stdin.end();
  await new Promise((r) => ff.on('close', r));
  console.log(`→ ${OUT} (${DUR}s, ${n} images)`);
}
await browser.close();
