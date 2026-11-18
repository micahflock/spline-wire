// Bundled by e2e/decode.spec.ts (esbuild) into a test page: the pipeline's
// browser decoding, detection and measuring, on globalThis.
import { specFromDict } from "../src/pipeline/chain";
import { decodeToGray } from "../src/pipeline/decode";
import { detectFiducials } from "../src/pipeline/detect";
import { measureDetections } from "../src/pipeline/measure";

(globalThis as Record<string, unknown>).pipeline = { specFromDict, decodeToGray, detectFiducials, measureDetections };
