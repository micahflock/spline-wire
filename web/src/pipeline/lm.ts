// Levenberg–Marquardt least squares: a port of MINPACK's lmder (with lmpar,
// qrfac, qrsolv and enorm), as SciPy's least_squares(method="lm") runs it.
// SciPy (1.16 and later) passes lmder a Jacobian by forward differences
// (_numdiff.approx_derivative, "2-point"), diag=None (mode 1: variables
// scaled by the Jacobian's column norms) and factor 100. Its defaults are
// kept: ftol = xtol = gtol = 1e-8. The steps follow MINPACK's, so the same
// problem converges along the same path to the same point, up to rounding.

const EPSMCH = 2.220446049250313e-16;
const DWARF = 2.2250738585072014e-308;

export interface LeastSquaresResult {
  x: number[];
  fun: number[];
  cost: number;          // 0.5 * sum(fun^2), as SciPy reports it
  nfev: number;
  status: number;        // MINPACK's info: 1-4 converged, 5 too many evaluations, 6-8 tolerances too small
}

export interface LeastSquaresOptions {
  maxNfev?: number;      // evaluations, not counting the Jacobian's; default 100 * n, as SciPy's lm
  ftol?: number;
  xtol?: number;
  gtol?: number;
}

/** Minimise sum(fun(x)^2) from x0. Throws a ValueError-like Error, as SciPy
 * does, when the residuals at x0 are not finite. */
export function leastSquaresLM(
  fun: (x: readonly number[]) => number[],
  x0: readonly number[],
  opts: LeastSquaresOptions = {},
): LeastSquaresResult {
  const n = x0.length;
  const ftol = opts.ftol ?? 1e-8;
  const xtol = opts.xtol ?? 1e-8;
  const gtol = opts.gtol ?? 1e-8;
  const maxfev = opts.maxNfev ?? 100 * n;
  const factor = 100;

  let x = x0.slice();
  let fvec = fun(x);
  const m = fvec.length;
  if (!fvec.every(Number.isFinite)) throw new NotFinite();
  if (m < n) throw new Error(`least squares needs at least as many residuals (${m}) as unknowns (${n})`);
  let nfev = 1;

  // fjac[i][j], row i, column j, as MINPACK's fjac(i, j).
  let fjac: number[][] = [];
  const diag = new Array<number>(n).fill(0);
  const qtf = new Array<number>(n).fill(0);
  const ipvt = new Array<number>(n).fill(0);
  let fnorm = enorm(fvec);
  let par = 0;
  let iter = 1;
  let info = 0;
  let delta = 0;
  let xnorm = 0;

  outer: for (;;) {
    fjac = jacobian(fun, x, fvec);
    const { rdiag, acnorm } = qrfac(m, n, fjac, ipvt);
    if (iter === 1) {
      for (let j = 0; j < n; j++) diag[j] = acnorm[j] === 0 ? 1 : acnorm[j]!;
      xnorm = enorm(x.map((xj, j) => diag[j]! * xj));
      delta = factor * xnorm;
      if (delta === 0) delta = factor;
    }
    // Q^T fvec; its first n entries in qtf.
    const wa4 = fvec.slice();
    for (let j = 0; j < n; j++) {
      if (fjac[j]![j] !== 0) {
        let sum = 0;
        for (let i = j; i < m; i++) sum += fjac[i]![j]! * wa4[i]!;
        const temp = -sum / fjac[j]![j]!;
        for (let i = j; i < m; i++) wa4[i] = wa4[i]! + fjac[i]![j]! * temp;
      }
      fjac[j]![j] = rdiag[j]!;
      qtf[j] = wa4[j]!;
    }
    // Norm of the scaled gradient.
    let gnorm = 0;
    if (fnorm !== 0) {
      for (let j = 0; j < n; j++) {
        const l = ipvt[j]!;
        if (acnorm[l] !== 0) {
          let sum = 0;
          for (let i = 0; i <= j; i++) sum += fjac[i]![j]! * (qtf[i]! / fnorm);
          gnorm = Math.max(gnorm, Math.abs(sum / acnorm[l]!));
        }
      }
    }
    if (gnorm <= gtol) info = 4;
    if (info !== 0) break;
    for (let j = 0; j < n; j++) diag[j] = Math.max(diag[j]!, acnorm[j]!);

    for (;;) {
      const lp = lmpar(n, fjac, ipvt, diag, qtf, delta, par);
      par = lp.par;
      const step = lp.x.map((v) => -v);
      const xNew = x.map((xj, j) => xj + step[j]!);
      const pnorm = enorm(step.map((s, j) => diag[j]! * s));
      if (iter === 1) delta = Math.min(delta, pnorm);
      const fNew = fun(xNew);
      nfev++;
      const fnorm1 = enorm(fNew);
      let actred = -1;
      if (0.1 * fnorm1 < fnorm) actred = 1 - (fnorm1 / fnorm) ** 2;
      // The predicted reduction and the directional derivative.
      const wa3 = new Array<number>(n).fill(0);
      for (let j = 0; j < n; j++) {
        const temp = step[ipvt[j]!]!;
        for (let i = 0; i <= j; i++) wa3[i] = wa3[i]! + fjac[i]![j]! * temp;
      }
      const temp1 = enorm(wa3) / fnorm;
      const temp2 = (Math.sqrt(par) * pnorm) / fnorm;
      const prered = temp1 * temp1 + (temp2 * temp2) / 0.5;
      const dirder = -(temp1 * temp1 + temp2 * temp2);
      const ratio = prered !== 0 ? actred / prered : 0;
      if (ratio <= 0.25) {
        let temp = actred >= 0 ? 0.5 : (0.5 * dirder) / (dirder + 0.5 * actred);
        if (0.1 * fnorm1 >= fnorm || temp < 0.1) temp = 0.1;
        delta = temp * Math.min(delta, pnorm / 0.1);
        par = par / temp;
      } else if (par === 0 || ratio >= 0.75) {
        delta = pnorm / 0.5;
        par = 0.5 * par;
      }
      if (ratio >= 1e-4) {
        x = xNew;
        fvec = fNew;
        xnorm = enorm(x.map((xj, j) => diag[j]! * xj));
        fnorm = fnorm1;
        iter++;
      }
      if (Math.abs(actred) <= ftol && prered <= ftol && 0.5 * ratio <= 1) info = 1;
      if (delta <= xtol * xnorm) info = 2;
      if (Math.abs(actred) <= ftol && prered <= ftol && 0.5 * ratio <= 1 && info === 2) info = 3;
      if (info !== 0) break outer;
      if (nfev >= maxfev) info = 5;
      if (Math.abs(actred) <= EPSMCH && prered <= EPSMCH && 0.5 * ratio <= 1) info = 6;
      if (delta <= EPSMCH * xnorm) info = 7;
      if (gnorm <= EPSMCH) info = 8;
      if (info !== 0) break outer;
      if (ratio >= 1e-4) break;
    }
  }
  let cost = 0;
  for (const f of fvec) cost += f * f;
  return { x, fun: fvec, cost: 0.5 * cost, nfev, status: info };
}

/** SciPy's error for residuals that are not finite at the start. */
export class NotFinite extends Error {
  constructor() {
    super("Residuals are not finite in the initial point.");
  }
}

/** Forward differences as scipy.optimize._numdiff.approx_derivative
 * ("2-point", default step): h = sqrt(eps) * sign(x) * max(1, |x|), with
 * sign(0) = 1, and the step actually taken, (x + h) - x. */
function jacobian(fun: (x: readonly number[]) => number[], x: readonly number[], f0: readonly number[]): number[][] {
  const n = x.length;
  const m = f0.length;
  const rel = Math.sqrt(EPSMCH);
  const J: number[][] = Array.from({ length: m }, () => new Array<number>(n).fill(0));
  for (let j = 0; j < n; j++) {
    const xj = x[j]!;
    const h = rel * (xj >= 0 ? 1 : -1) * Math.max(1, Math.abs(xj));
    const x1 = x.slice();
    x1[j] = xj + h;
    const dx = x1[j]! - xj;
    const f1 = fun(x1);
    for (let i = 0; i < m; i++) J[i]![j] = (f1[i]! - f0[i]!) / dx;
  }
  return J;
}

/** MINPACK's enorm: the Euclidean norm, guarded against overflow and underflow. */
export function enorm(v: readonly number[]): number {
  const rdwarf = 3.834e-20;
  const rgiant = 1.304e19;
  let s1 = 0, s2 = 0, s3 = 0, x1max = 0, x3max = 0;
  const agiant = rgiant / v.length;
  for (const vi of v) {
    const xabs = Math.abs(vi);
    if (xabs > rdwarf && xabs < agiant) {
      s2 += xabs * xabs;
    } else if (xabs <= rdwarf) {
      if (xabs > x3max) {
        s3 = 1 + s3 * (x3max / xabs) ** 2;
        x3max = xabs;
      } else if (xabs !== 0) {
        s3 += (xabs / x3max) ** 2;
      }
    } else if (xabs > x1max) {
      s1 = 1 + s1 * (x1max / xabs) ** 2;
      x1max = xabs;
    } else {
      s1 += (xabs / x1max) ** 2;
    }
  }
  if (s1 !== 0) return x1max * Math.sqrt(s1 + s2 / x1max / x1max);
  if (s2 !== 0) {
    return s2 >= x3max ? Math.sqrt(s2 * (1 + (x3max / s2) * (x3max * s3))) : Math.sqrt(x3max * (s2 / x3max + x3max * s3));
  }
  return x3max * Math.sqrt(s3);
}

/** QR factorisation with column pivoting, in place (MINPACK qrfac, pivot on). */
function qrfac(m: number, n: number, a: number[][], ipvt: number[]): { rdiag: number[]; acnorm: number[] } {
  const acnorm = new Array<number>(n);
  const rdiag = new Array<number>(n);
  const wa = new Array<number>(n);
  for (let j = 0; j < n; j++) {
    const col: number[] = [];
    for (let i = 0; i < m; i++) col.push(a[i]![j]!);
    acnorm[j] = enorm(col);
    rdiag[j] = acnorm[j]!;
    wa[j] = rdiag[j]!;
    ipvt[j] = j;
  }
  const minmn = Math.min(m, n);
  for (let j = 0; j < minmn; j++) {
    // Bring the column of largest norm into the pivot position.
    let kmax = j;
    for (let k = j; k < n; k++) if (rdiag[k]! > rdiag[kmax]!) kmax = k;
    if (kmax !== j) {
      for (let i = 0; i < m; i++) {
        const t = a[i]![j]!;
        a[i]![j] = a[i]![kmax]!;
        a[i]![kmax] = t;
      }
      rdiag[kmax] = rdiag[j]!;
      wa[kmax] = wa[j]!;
      const k = ipvt[j]!;
      ipvt[j] = ipvt[kmax]!;
      ipvt[kmax] = k;
    }
    // The Householder transformation that reduces column j to a multiple of e_j.
    const colj: number[] = [];
    for (let i = j; i < m; i++) colj.push(a[i]![j]!);
    let ajnorm = enorm(colj);
    if (ajnorm !== 0) {
      if (a[j]![j]! < 0) ajnorm = -ajnorm;
      for (let i = j; i < m; i++) a[i]![j] = a[i]![j]! / ajnorm;
      a[j]![j] = a[j]![j]! + 1;
      for (let k = j + 1; k < n; k++) {
        let sum = 0;
        for (let i = j; i < m; i++) sum += a[i]![j]! * a[i]![k]!;
        const temp = sum / a[j]![j]!;
        for (let i = j; i < m; i++) a[i]![k] = a[i]![k]! - temp * a[i]![j]!;
        if (rdiag[k] !== 0) {
          const t = a[j]![k]! / rdiag[k]!;
          rdiag[k] = rdiag[k]! * Math.sqrt(Math.max(0, 1 - t * t));
          if (0.05 * (rdiag[k]! / wa[k]!) ** 2 <= EPSMCH) {
            const rest: number[] = [];
            for (let i = j + 1; i < m; i++) rest.push(a[i]![k]!);
            rdiag[k] = enorm(rest);
            wa[k] = rdiag[k]!;
          }
        }
      }
    }
    rdiag[j] = -ajnorm;
  }
  return { rdiag, acnorm };
}

/** The Levenberg–Marquardt parameter for a step within delta (MINPACK lmpar). */
function lmpar(
  n: number, r: number[][], ipvt: readonly number[], diag: readonly number[], qtb: readonly number[],
  delta: number, par0: number,
): { par: number; x: number[]; sdiag: number[] } {
  let par = par0;
  const x = new Array<number>(n).fill(0);
  let sdiag = new Array<number>(n).fill(0);
  const wa1 = new Array<number>(n).fill(0);
  const wa2 = new Array<number>(n).fill(0);
  // The Gauss–Newton direction; least squares if the Jacobian is rank-deficient.
  let nsing = n;
  for (let j = 0; j < n; j++) {
    wa1[j] = qtb[j]!;
    if (r[j]![j] === 0 && nsing === n) nsing = j;
    if (nsing < n) wa1[j] = 0;
  }
  for (let k = 0; k < nsing; k++) {
    const j = nsing - 1 - k;
    wa1[j] = wa1[j]! / r[j]![j]!;
    const temp = wa1[j]!;
    for (let i = 0; i < j; i++) wa1[i] = wa1[i]! - r[i]![j]! * temp;
  }
  for (let j = 0; j < n; j++) x[ipvt[j]!] = wa1[j]!;

  let iter = 0;
  for (let j = 0; j < n; j++) wa2[j] = diag[j]! * x[j]!;
  let dxnorm = enorm(wa2);
  let fp = dxnorm - delta;
  if (fp <= 0.1 * delta) return { par: 0, x, sdiag };

  // Lower bound parl for the zero of the function, if the Jacobian has full rank.
  let parl = 0;
  if (nsing >= n) {
    for (let j = 0; j < n; j++) {
      const l = ipvt[j]!;
      wa1[j] = diag[l]! * (wa2[l]! / dxnorm);
    }
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let i = 0; i < j; i++) sum += r[i]![j]! * wa1[i]!;
      wa1[j] = (wa1[j]! - sum) / r[j]![j]!;
    }
    const temp = enorm(wa1);
    parl = fp / delta / temp / temp;
  }
  // Upper bound paru.
  for (let j = 0; j < n; j++) {
    let sum = 0;
    for (let i = 0; i <= j; i++) sum += r[i]![j]! * qtb[i]!;
    wa1[j] = sum / diag[ipvt[j]!]!;
  }
  const gnorm = enorm(wa1);
  let paru = gnorm / delta;
  if (paru === 0) paru = DWARF / Math.min(delta, 0.1);
  par = Math.max(par, parl);
  par = Math.min(par, paru);
  if (par === 0) par = gnorm / dxnorm;

  for (;;) {
    iter++;
    if (par === 0) par = Math.max(DWARF, 0.001 * paru);
    const temp = Math.sqrt(par);
    for (let j = 0; j < n; j++) wa1[j] = temp * diag[j]!;
    sdiag = qrsolv(n, r, ipvt, wa1, qtb, x);
    for (let j = 0; j < n; j++) wa2[j] = diag[j]! * x[j]!;
    dxnorm = enorm(wa2);
    const prev = fp;
    fp = dxnorm - delta;
    if (Math.abs(fp) <= 0.1 * delta || (parl === 0 && fp <= prev && prev < 0) || iter === 10) break;
    // Newton's correction.
    for (let j = 0; j < n; j++) {
      const l = ipvt[j]!;
      wa1[j] = diag[l]! * (wa2[l]! / dxnorm);
    }
    for (let j = 0; j < n; j++) {
      wa1[j] = wa1[j]! / sdiag[j]!;
      const t = wa1[j]!;
      for (let i = j + 1; i < n; i++) wa1[i] = wa1[i]! - r[i]![j]! * t;
    }
    const t = enorm(wa1);
    const parc = fp / delta / t / t;
    if (fp > 0) parl = Math.max(parl, par);
    if (fp < 0) paru = Math.min(paru, par);
    par = Math.max(parl, par + parc);
  }
  return { par, x, sdiag };
}

/** Solve R z = Q^T b in the least-squares sense with the diagonal D appended
 * (MINPACK qrsolv). Writes x; returns the diagonal of the new triangle. */
function qrsolv(
  n: number, r: number[][], ipvt: readonly number[], diag: readonly number[], qtb: readonly number[], x: number[],
): number[] {
  const sdiag = new Array<number>(n).fill(0);
  const wa = new Array<number>(n).fill(0);
  for (let j = 0; j < n; j++) {
    for (let i = j; i < n; i++) r[i]![j] = r[j]![i]!;
    x[j] = r[j]![j]!;
    wa[j] = qtb[j]!;
  }
  for (let j = 0; j < n; j++) {
    const l = ipvt[j]!;
    if (diag[l] !== 0) {
      for (let k = j; k < n; k++) sdiag[k] = 0;
      sdiag[j] = diag[l]!;
      let qtbpj = 0;
      for (let k = j; k < n; k++) {
        if (sdiag[k] === 0) continue;
        let sin: number, cos: number;
        if (Math.abs(r[k]![k]!) < Math.abs(sdiag[k]!)) {
          const cotan = r[k]![k]! / sdiag[k]!;
          sin = 0.5 / Math.sqrt(0.25 + 0.25 * cotan * cotan);
          cos = sin * cotan;
        } else {
          const tan = sdiag[k]! / r[k]![k]!;
          cos = 0.5 / Math.sqrt(0.25 + 0.25 * tan * tan);
          sin = cos * tan;
        }
        r[k]![k] = cos * r[k]![k]! + sin * sdiag[k]!;
        const temp = cos * wa[k]! + sin * qtbpj;
        qtbpj = -sin * wa[k]! + cos * qtbpj;
        wa[k] = temp;
        for (let i = k + 1; i < n; i++) {
          const t = cos * r[i]![k]! + sin * sdiag[i]!;
          sdiag[i] = -sin * r[i]![k]! + cos * sdiag[i]!;
          r[i]![k] = t;
        }
      }
    }
    sdiag[j] = r[j]![j]!;
    r[j]![j] = x[j]!;
  }
  let nsing = n;
  for (let j = 0; j < n; j++) {
    if (sdiag[j] === 0 && nsing === n) nsing = j;
    if (nsing < n) wa[j] = 0;
  }
  for (let k = 0; k < nsing; k++) {
    const j = nsing - 1 - k;
    let sum = 0;
    for (let i = j + 1; i < nsing; i++) sum += r[i]![j]! * wa[i]!;
    wa[j] = (wa[j]! - sum) / sdiag[j]!;
  }
  for (let j = 0; j < n; j++) x[ipvt[j]!] = wa[j]!;
  return sdiag;
}
