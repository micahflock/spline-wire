/** The pipeline's message for a person, Python's ValueError: shown as is.
 * Anything else thrown from the pipeline is a bug. */
export class PipelineError extends Error {
  override name = "PipelineError";
}
