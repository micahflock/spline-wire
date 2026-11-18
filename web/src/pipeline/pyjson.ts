// json.dumps as Python writes it, for files that must match Python's byte for
// byte (output.json_text): floats in repr form ("1.0", "1e-05", "-0.0"),
// non-ASCII escaped as \uXXXX, ", " and ": " separators with an indent.
// JavaScript has one number type, so floats are marked: wrap a number in
// pyFloat() where Python holds a float; a bare number is written as an int.

export class PyFloat {
  constructor(readonly value: number) {}
}

export function pyFloat(x: number): PyFloat {
  return new PyFloat(x);
}

/** repr(float). */
export function floatRepr(x: number): string {
  if (Number.isNaN(x)) return "NaN";
  if (x === Infinity) return "Infinity";
  if (x === -Infinity) return "-Infinity";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  // The shortest digits that read back as x, and the decimal exponent.
  const [mant, expStr] = x.toExponential().split("e") as [string, string];
  const exp = Number(expStr);
  const neg = mant.startsWith("-");
  const digits = mant.replace("-", "").replace(".", "");
  let out: string;
  if (exp < -4 || exp >= 16) {
    const m = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    out = `${m}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
  } else if (exp < 0) {
    out = `0.${"0".repeat(-exp - 1)}${digits}`;
  } else if (digits.length <= exp + 1) {
    out = `${digits}${"0".repeat(exp + 1 - digits.length)}.0`;
  } else {
    out = `${digits.slice(0, exp + 1)}.${digits.slice(exp + 1)}`;
  }
  return neg ? `-${out}` : out;
}

function str(s: string): string {
  let out = '"';
  for (const ch of s) {
    const c = ch.codePointAt(0)!;
    if (ch === '"') out += '\\"';
    else if (ch === "\\") out += "\\\\";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (ch === "\b") out += "\\b";
    else if (ch === "\f") out += "\\f";
    else if (c < 0x20 || c > 0x7e) {
      if (c > 0xffff) {
        const v = c - 0x10000;
        out += `\\u${(0xd800 + (v >> 10)).toString(16)}\\u${(0xdc00 + (v & 0x3ff)).toString(16)}`;
      } else {
        out += `\\u${c.toString(16).padStart(4, "0")}`;
      }
    } else out += ch;
  }
  return out + '"';
}

/** json.dumps(value, indent=indent). */
export function pyDumps(value: unknown, indent = 2, level = 0): string {
  if (value === null || value === undefined) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (value instanceof PyFloat) return floatRepr(value.value);
  if (typeof value === "number") {
    if (!Number.isInteger(value)) throw new Error(`pyDumps: ${value} is not an int; wrap floats in pyFloat()`);
    return String(value);
  }
  if (typeof value === "string") return str(value);
  const pad = "\n" + " ".repeat(indent * (level + 1));
  const end = "\n" + " ".repeat(indent * level);
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    return "[" + value.map((v) => pad + pyDumps(v, indent, level + 1)).join(",") + end + "]";
  }
  const entries = Object.entries(value as Record<string, unknown>);
  if (!entries.length) return "{}";
  return "{" + entries.map(([k, v]) => `${pad}${str(k)}: ${pyDumps(v, indent, level + 1)}`).join(",") + end + "}";
}
