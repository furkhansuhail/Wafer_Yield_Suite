#!/usr/bin/env node
/**
 * scripts/run.mjs
 * ===============
 * Launches the Python pieces from the project-local venv created by bootstrap.
 *
 *   node scripts/run.mjs dashboard          streamlit run dashboard/streamlit_app.py
 *   node scripts/run.mjs server [--http]    python -m mcp_server.server
 *   node scripts/run.mjs train <domain>     fit a model (secom | yield_curve)
 *
 * Add --check to print the resolved command without executing it.
 */
import { spawn } from "node:child_process";
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

const err = (m) => {
  console.error(`\x1b[31m[run]\x1b[0m ${m}`);
  process.exit(1);
};

const argv = process.argv.slice(2);
const check = argv.includes("--check");
const args = argv.filter((a) => a !== "--check");
const mode = args[0];

if (!check && !existsSync(venvPython)) {
  err("Python venv not found. Run `npm run setup` first (or `npm install`).");
}

let cmd, cmdArgs, cwd = PLATFORM;

if (mode === "dashboard" || mode === undefined) {
  // `python -m streamlit run` is more portable than the streamlit console script.
  cmd = venvPython;
  cmdArgs = ["-m", "streamlit", "run", "dashboard/streamlit_app.py", ...args.slice(1)];
} else if (mode === "server") {
  cmd = venvPython;
  cmdArgs = ["-m", "mcp_server.server", ...args.slice(1)]; // pass-through e.g. --http
} else if (mode === "train") {
  const domain = args[1];
  if (!["secom", "yield_curve"].includes(domain)) {
    err("Usage: run.mjs train <secom|yield_curve>  (CNN training is a CLI job; see README)");
  }
  const fn = domain === "secom" ? "_train_secom" : "_train_yield";
  const py = [
    "import sys, json",
    "sys.path.insert(0, '.')",
    `from mcp_server.server import ${fn} as _t`,
    "print(json.dumps(_t(), indent=2))",
  ].join("; ");
  cmd = venvPython;
  cmdArgs = ["-c", py];
} else {
  err(`Unknown mode '${mode}'. Use: dashboard | server | train`);
}

if (check) {
  console.log(JSON.stringify({ cwd, cmd, args: cmdArgs }, null, 2));
  process.exit(0);
}

const child = spawn(cmd, cmdArgs, { stdio: "inherit", cwd });
child.on("exit", (code) => process.exit(code ?? 0));
child.on("error", (e) => err(`Failed to launch: ${e.message}`));
