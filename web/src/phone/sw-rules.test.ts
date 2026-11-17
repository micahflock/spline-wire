import { describe, expect, it } from "vitest";
import { cacheName, cacheable, parseManifest, routeOf, staleCaches, staleEntries, type CoreManifest } from "./sw-rules";

const scope = "https://example.org/spline-wire/";
const manifest: CoreManifest = {
  pyodide: { version: "314.0.8", index: "pyodide/" },
  wheels: { splinewire: "wheels/3c587e3156ec/spline_wire-0.7.0-py3-none-any.whl", ezdxf: "wheels/ezdxf-1.4.4-py3-none-any.whl" },
};

describe("service worker rules", () => {
  it("caches only the core's assets, and only GETs under its scope", () => {
    expect(routeOf(scope + "pyodide/pyodide.asm.wasm", scope)).toBe("core");
    expect(routeOf(scope + "wheels/ezdxf-1.4.4-py3-none-any.whl", scope)).toBe("core");
    expect(routeOf(scope + "core-manifest.json?x=1", scope)).toBe("manifest");
    for (const url of [scope + "phone/", scope + "assets/phone-abc.js", scope + "chains.json", "https://cdn.example/pyodide/x.whl"]) {
      expect(routeOf(url, scope)).toBe("other");
    }
    expect(routeOf(scope + "pyodide/pyodide.mjs", scope, "POST")).toBe("other");
  });

  it("names the cache after Pyodide's version, and drops other versions and layouts", () => {
    const name = cacheName(manifest);
    expect(name).toBe("splinewire-core-1-pyodide-314.0.8");
    expect(staleCaches([name, "splinewire-core-1-pyodide-0.29.0", "splinewire-core-0-pyodide-314.0.8", "splinewire-meta-1",
      "splinewire-meta-0", "someone-else"], name))
      .toEqual(["splinewire-core-1-pyodide-0.29.0", "splinewire-core-0-pyodide-314.0.8", "splinewire-meta-0"]);
  });

  it("drops wheels the manifest no longer lists, keeping Pyodide's files", () => {
    const urls = [
      scope + "pyodide/pyodide.asm.wasm",
      scope + "wheels/3c587e3156ec/spline_wire-0.7.0-py3-none-any.whl",
      scope + "wheels/0123456789ab/spline_wire-0.6.0-py3-none-any.whl",
      scope + "wheels/ezdxf-1.4.4-py3-none-any.whl",
    ];
    expect(staleEntries(urls, scope, manifest)).toEqual([scope + "wheels/0123456789ab/spline_wire-0.6.0-py3-none-any.whl"]);
  });

  it("reads a manifest, refusing anything else", () => {
    expect(parseManifest(JSON.stringify(manifest))).toEqual(manifest);
    expect(parseManifest("<html>")).toBeNull();
    expect(parseManifest(JSON.stringify({ pyodide: {} }))).toBeNull();
  });

  it("keeps whole answers only", () => {
    expect(cacheable(200, "basic")).toBe(true);
    expect(cacheable(206, "basic")).toBe(false);
    expect(cacheable(404, "basic")).toBe(false);
    expect(cacheable(0, "opaque")).toBe(false);
  });
});
