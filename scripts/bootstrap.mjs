#!/usr/bin/env node
/**
 * scripts/bootstrap.mjs
 * =====================
 * Runs on `npm install` (via the postinstall hook) and on `npm run setup`.
 *
 * npm cannot install Python packages, so this script bridges the gap: it finds
 * a Python 3.10+ interpreter, creates a project-local virtualenv at ./.venv, and
 * pip-installs the platform requirements (MCP server + Streamlit) into it. After
 * this, `npm start` launches Streamlit from that venv.
 *
 * Env switches:
 *   WAFER_SKIP_BOOTSTRAP=1  skip entirely (useful in CI / when you manage Python yourself)
 *   WAFER_WITH_TRAIN=1      also install the training extras (sklearn, scipy, ...)
 *   WAFER_PYTHON=/path/python  force a specific interpreter
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const PLATFORM = join(ROOT, "wafer_mcp_platform");
const VENV = join(ROOT, ".venv");
const isWin = process.platform === "win32";
const venvPython = isWin
  ? join(VENV, "Scripts", "python.exe")
  : join(VENV, "bin", "python");

const log = (m) => console.log(`\x1b[36m[bootstrap]\x1b[0m ${m}`);
const warn = (m) => console.warn(`\x1b[33m[bootstrap]\x1b[0m ${m}`);
const fail = (m) => {
  console.error(`\x1b[31m[bootstrap]\x1b[0m ${m}`);
  process.exit(1);
};

if (process.env.WAFER_SKIP_BOOTSTRAP === "1") {
  log("WAFER_SKIP_BOOTSTRAP=1 set — skipping Python setup.");
  process.exit(0);
}

function run(cmd, args, opts = {}) {
  return spawnSync(cmd, args, { stdio: "inherit", encoding: "utf8", ...opts });
}

function pythonVersionOK(cmd) {
  const r = spawnSync(cmd, ["-c", "import sys;print('%d.%d'%sys.version_info[:2])"], {
    encoding: "utf8",
  });
  if (r.status !== 0 || !r.stdout) return null;
  const [maj, min] = r.stdout.trim().split(".").map(Number);
  return maj === 3 && min >= 10 ? r.stdout.trim() : null;
}

function findPython() {
  const candidates = [
    process.env.WAFER_PYTHON,
    "python3",
    "python",
    isWin ? "py" : null,
  ].filter(Boolean);
  for (const c of candidates) {
    const v = pythonVersionOK(c);
    if (v) return { cmd: c, version: v };
  }
  return null;
}

// 1) locate a usable Python ------------------------------------------------- #
const py = findPython();
if (!py) {
  fail(
    "No Python 3.10+ found on PATH.\n" +
      "  Install Python 3.10 or newer, then re-run:  npm run setup\n" +
      "  (or set WAFER_PYTHON=/path/to/python and re-run)"
  );
}
log(`Using Python ${py.version} (${py.cmd})`);

// 2) create the virtualenv -------------------------------------------------- #
if (existsSync(venvPython)) {
  log(".venv already exists — reusing it.");
} else {
  log("Creating virtualenv at ./.venv ...");
  // --copies avoids symlinks, which some filesystems (network mounts, certain
  // container volumes) refuse to create.
  const r = run(py.cmd, ["-m", "venv", "--copies", VENV]);
  if (r.status !== 0) fail("Failed to create the virtualenv (is the venv module available?).");
}

// 3) install Python dependencies into the venv ------------------------------ #
log("Upgrading pip ...");
run(venvPython, ["-m", "pip", "install", "--upgrade", "pip", "-q"]);

log("Installing platform requirements (MCP server + Streamlit) ...");
let r = run(venvPython, [
  "-m", "pip", "install", "-q",
  "-r", join(PLATFORM, "requirements.txt"),
]);
if (r.status !== 0) fail("pip install of platform requirements failed (see output above).");

if (process.env.WAFER_WITH_TRAIN === "1" || process.argv.includes("--with-train")) {
  log("Installing training extras (sklearn, imbalanced-learn, scipy, ucimlrepo) ...");
  r = run(venvPython, [
    "-m", "pip", "install", "-q",
    "-r", join(ROOT, "requirements-train.txt"),
  ]);
  if (r.status !== 0) warn("Training extras failed to install; `npm run train:*` may not work.");
}

log("Done. Next:");
log("  npm start              # launch the Streamlit dashboard (+ MCP server)");
log("  npm run server         # run the MCP server alone (stdio)");
log("  npm run train:yield    # fit the yield models (needs: npm run setup:train)");
