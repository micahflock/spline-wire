import { defineConfig } from "vitest/config";

// The relay's own tests: the rules (also run by web/'s npm test) and the
// integration tests with real sockets, which need this folder's packages.
export default defineConfig({
  test: {
    include: ["*.test.ts", "test/*.itest.ts"],
    environment: "node",
    testTimeout: 60_000,
    fileParallelism: false,
  },
});
