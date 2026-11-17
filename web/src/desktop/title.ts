// The tab's title counts curves that arrived while the tab was hidden, so a
// person working in Fusion sees "(1) Spline Wire" on the tab and knows the
// curve is there. Seeing the page clears it.

export class UnseenTitle {
  private count = 0;

  constructor(private readonly base: string) {}

  /** A curve arrived. */
  arrived(hidden: boolean): void {
    if (hidden) this.count++;
  }

  /** The page became visible. */
  seen(): void {
    this.count = 0;
  }

  get unseen(): number {
    return this.count;
  }

  get title(): string {
    return this.count ? `(${this.count}) ${this.base}` : this.base;
  }
}
