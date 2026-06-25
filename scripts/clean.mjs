#!/usr/bin/env node
/** scripts/clean.mjs — remove the venv, the data/model workspace, and caches. */
import { rmSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { execSync } from "node:child_process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

for (const p of [".venv", join("wafer_mcp_platform", "_workspace")]) {
  try {
    rmSync(join(ROOT, p), { recursive: true, force: true });
    console.log(`[clean] removed ${p}`);
  } catch {}
}
// best-effort __pycache__ sweep
try {
  const cmd = process.platform === "win32"
    ? `for /d /r "${ROOT}" %d in (__pycache__) do @rmdir /s /q "%d"`
    : `find "${ROOT}" -name __pycache__ -type d -prune -exec rm -rf {} +`;
  execSync(cmd, { stdio: "ignore", shell: true });
} catch {}
console.log("[clean] done");
