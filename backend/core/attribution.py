"""Where a security's factor risk comes from, by block and by factor.

A covariance matrix on its own has no risk contributions. It has variances,
correlations and eigenvalues, and every one of those is a statement about the
factors rather than about anything anyone holds. "Which block drives the risk"
only becomes a question once there is an exposure vector to drive it, which is
why this module takes betas and why the page that uses it needs a security.

Everything here is exact rather than approximate, and for one reason. Volatility
is homogeneous of degree one in the exposures, so by Euler's theorem

    sigma = sum_i b_i * d(sigma)/d(b_i)

holds identically, not to a tolerance. That turns "this block is 40% of the
risk" from a figure of speech into arithmetic that adds up, and the tests assert
the identities to machine precision.

Three numbers are produced and they answer three different questions.

*Contribution* is what a block adds given everything else the security is
exposed to. It is the additive one: the block contributions sum to the total,
and this is the column to read when asking which block to look at first.

*Standalone* is what a block's exposures would produce on their own, with the
rest of the book switched off. It does not add up, and it is not supposed to:
the gap between the sum of the standalones and the total is the diversification
across blocks, which is itself worth seeing.

*The block-by-block variance matrix* splits total variance into own-block terms
on the diagonal and cross-block covariance off it. A clean block structure is
diagonally dominant. Large off-diagonals mean the blocks are sharing the same
underlying moves, which is a statement about the factor design rather than about
the security, and the orthogonalisation is what is supposed to keep them small.

The three are consistent by construction: each row of the variance matrix sums
to that block's contribution times the total, so the bar chart is the row sums
of the heatmap. That is asserted too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Below this the security has no meaningful factor risk and every share becomes
# a ratio of two numbers that are both noise.
_FLOOR = 1e-12


@dataclass
class BlockAttribution:
    """One security's factor risk, split by block and by factor within it."""

    factors: list[str]
    factor_block: list[str]
    blocks: list[str]

    sigma: float
    """Factor volatility, sqrt(b' S b), in the units of the covariance."""

    mctr_factor: np.ndarray
    """d(sigma)/d(b_i). What one more unit of this exposure would cost."""

    ctr_factor: np.ndarray
    """b_i * mctr_i. Sums to sigma exactly. Negative where a factor hedges."""

    ctr_block: np.ndarray
    """Per block, summed over its factors. Sums to sigma exactly."""

    standalone_block: np.ndarray
    """sqrt(b_b' S_bb b_b). What the block would produce alone. Not additive."""

    variance_block: np.ndarray
    """Block by block. Diagonal is own variance, off-diagonal cross-block
    covariance; the whole matrix sums to sigma squared."""

    @property
    def pct_factor(self) -> np.ndarray:
        """Each factor's share of total risk. Signed, and can exceed one."""
        return self.ctr_factor / self.sigma if self.sigma > _FLOOR else \
            np.zeros_like(self.ctr_factor)

    @property
    def pct_block(self) -> np.ndarray:
        return self.ctr_block / self.sigma if self.sigma > _FLOOR else \
            np.zeros_like(self.ctr_block)

    @property
    def pct_variance_block(self) -> np.ndarray:
        """Each block's share of total variance.

        Identical to `pct_block`, and that is worth knowing rather than a
        coincidence: the contribution share in volatility units is
        b_b'(Sb)_b / sigma^2, which is already a share of variance. The two
        readings of "40% of the risk" agree here, so the page does not have to
        say which one it means.
        """
        return self.pct_block

    @property
    def block_correlation(self) -> np.ndarray:
        """Correlation between the blocks' own return streams.

        Each block's exposures define a sub-portfolio b_b'f_b, and this is the
        correlation matrix of those. Distinct from the variance matrix above: a
        block can be strongly correlated with another and still contribute
        almost nothing, if the exposure to it is small. Correlation says whether
        the diversification is available; contribution says whether it is being
        used.
        """
        sd = np.sqrt(np.clip(np.diag(self.variance_block), 0.0, None))
        outer = np.outer(sd, sd)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.where(outer > _FLOOR, self.variance_block / outer, 0.0)
        np.fill_diagonal(corr, np.where(sd > _FLOOR, 1.0, 0.0))
        return np.clip(corr, -1.0, 1.0)

    @property
    def undiversified(self) -> float:
        """The sum of the standalone block volatilities.

        What the security would be worth if the blocks never offset each other.
        Always at least sigma; the excess is what the offsetting is worth.
        """
        return float(np.sum(self.standalone_block))

    @property
    def diversification(self) -> float:
        """Volatility saved by holding the blocks together rather than apart."""
        return self.undiversified - self.sigma

    def within(self, block: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        """(factor ids, contributions, shares of the block) for one block.

        The share is of the block's own contribution, so it is signed and can
        exceed one: a block whose net contribution is small because two of its
        factors offset has shares that do not sit between zero and one, and
        drawing those as a pie would be a lie about the arithmetic.
        """
        idx = [i for i, b in enumerate(self.factor_block) if b == block]
        ctr = self.ctr_factor[idx]
        total = float(np.sum(ctr))
        share = ctr / total if abs(total) > _FLOOR else np.zeros_like(ctr)
        return [self.factors[i] for i in idx], ctr, share


def attribute(beta: np.ndarray, cov: np.ndarray, factors: list[str],
              factor_block: list[str]) -> BlockAttribution:
    """Split sqrt(beta' cov beta) across blocks and the factors inside them.

    `beta` and `cov` must describe the same factor series in the same order.
    Betas fitted on the orthogonalised panel paired with a covariance of the raw
    one is not a nuance but a different quantity: beta' Sigma beta stops being
    the variance of anything. The caller aligns them; this function trusts the
    order it is given and checks only the shapes.
    """
    beta = np.asarray(beta, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)
    k = beta.size
    if cov.shape != (k, k):
        raise ValueError(f"beta has {k} entries, covariance is {cov.shape}")
    if len(factors) != k or len(factor_block) != k:
        raise ValueError("factors and factor_block must line up with beta")
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(cov)):
        raise ValueError("beta and covariance must be finite")

    sigma_beta = cov @ beta
    variance = float(beta @ sigma_beta)
    # A covariance repaired to positive semi-definiteness can still return a
    # tiny negative here through rounding. Clipping at zero is right; carrying
    # the sign into a square root is not.
    sigma = float(np.sqrt(max(variance, 0.0)))

    if sigma <= _FLOOR:
        zeros = np.zeros(k)
        blocks = sorted(set(factor_block))
        return BlockAttribution(
            factors=list(factors), factor_block=list(factor_block), blocks=blocks,
            sigma=0.0, mctr_factor=zeros, ctr_factor=zeros,
            ctr_block=np.zeros(len(blocks)),
            standalone_block=np.zeros(len(blocks)),
            variance_block=np.zeros((len(blocks), len(blocks))))

    mctr = sigma_beta / sigma
    ctr = beta * mctr

    blocks = sorted(set(factor_block))
    members = {b: [i for i, fb in enumerate(factor_block) if fb == b]
               for b in blocks}

    ctr_block = np.array([float(np.sum(ctr[members[b]])) for b in blocks])
    standalone = np.array([
        float(np.sqrt(max(beta[members[b]] @ cov[np.ix_(members[b], members[b])]
                          @ beta[members[b]], 0.0)))
        for b in blocks])

    var_block = np.empty((len(blocks), len(blocks)))
    for r, b in enumerate(blocks):
        for c, d in enumerate(blocks):
            var_block[r, c] = float(
                beta[members[b]] @ cov[np.ix_(members[b], members[d])]
                @ beta[members[d]])
    # Symmetric by construction; the two multiplication orders differ in the
    # last bit or two. Averaging costs nothing and spares everything downstream
    # from having to decide which triangle to trust.
    var_block = (var_block + var_block.T) / 2.0

    return BlockAttribution(
        factors=list(factors), factor_block=list(factor_block), blocks=blocks,
        sigma=sigma, mctr_factor=mctr, ctr_factor=ctr, ctr_block=ctr_block,
        standalone_block=standalone, variance_block=var_block)


# ---------------------------------------------------------------------------
# what a block looks like from the inside, with no security involved
# ---------------------------------------------------------------------------

@dataclass
class BlockStructure:
    """One block's internal structure, read off the covariance alone.

    Nothing here depends on an exposure, which is the point: these answer "what
    is this block, and which factor speaks for it", not "what is it costing the
    security on screen". A block can be tightly coupled internally and still
    contribute nothing to a holding that has no exposure to it.
    """

    block: str
    factors: list[str]
    vol: np.ndarray
    """Each factor's own annualised volatility."""

    pc1_share: float
    """Share of the block's correlation trace explained by its first component.

    Correlation-based rather than covariance-based, so a single high-volatility
    factor cannot take the leading component through scale alone. Near one the
    block moves as one thing and a single factor can stand for it; near 1/k the
    factors inside it are close to unrelated and the block is a filing category
    rather than a driver.
    """

    pc1_loading: np.ndarray
    """Weights of the first eigenvector. Sign is arbitrary - an eigenvector
    times minus one is the same eigenvector - so it is oriented to make the
    largest weight positive, and read as grouping rather than direction."""

    centrality: np.ndarray
    """Mean correlation of each factor with the others in its block.

    The most central factor is the one that best represents the block; a factor
    near zero is in the block by economic classification but moves on its own,
    which is a finding about the factor set rather than about the market.
    """

    n_factors: int


def block_structure(cov: np.ndarray, factors: list[str], factor_block: list[str],
                    block: str) -> BlockStructure:
    """PCA, centrality and first-eigenvector weights for one block."""
    idx = [i for i, b in enumerate(factor_block) if b == block]
    sub = np.asarray(cov, dtype=float)[np.ix_(idx, idx)]
    names = [factors[i] for i in idx]
    k = len(idx)

    sd = np.sqrt(np.clip(np.diag(sub), 0.0, None))
    if k == 0:
        return BlockStructure(block, [], np.zeros(0), 0.0, np.zeros(0),
                              np.zeros(0), 0)
    if k == 1:
        # One factor is its own component and has nobody to correlate with.
        # Reporting 1.0 for the share is right; reporting 1.0 for centrality
        # would claim a relationship that has no second party.
        return BlockStructure(block, names, sd, 1.0, np.ones(1), np.zeros(1), 1)

    outer = np.outer(sd, sd)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(outer > _FLOOR, sub / outer, 0.0)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)

    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # The trace of a correlation matrix is k, so the share is the eigenvalue
    # over the number of factors rather than over the sum of eigenvalues, which
    # is the same thing and less prone to a rounding surprise.
    pc1_share = float(np.clip(vals[0] / k, 0.0, 1.0))

    loading = vecs[:, 0]
    if loading[np.argmax(np.abs(loading))] < 0:
        loading = -loading

    off = corr.copy()
    np.fill_diagonal(off, np.nan)
    centrality = np.nanmean(off, axis=1)

    return BlockStructure(block=block, factors=names, vol=sd,
                          pc1_share=pc1_share, pc1_loading=loading,
                          centrality=centrality, n_factors=k)


def stress_mask(panel: np.ndarray, quantile: float = 0.8,
                window: int = 21) -> np.ndarray:
    """Which days were stressed, judged by the panel's own common component.

    Regime has to be defined without reference to the block being studied, or
    the answer is circular: split the sample by equity volatility and the equity
    block will of course look different across the halves. So the split runs on
    the first principal component of the standardised panel - the move every
    factor shares - and calls the top slice of its rolling volatility stressed.

    Standardised first, so the classification is not decided by whichever factor
    happens to have the largest scale.

    Returns a boolean mask over rows. The caller re-estimates the covariance on
    each side; note that an EWMA cannot be used for that, because a scattered
    subset of days has no recency to weight by.
    """
    x = np.asarray(panel, dtype=float)
    if x.ndim != 2 or x.shape[0] < window * 2:
        return np.zeros(x.shape[0], dtype=bool)

    sd = x.std(axis=0, ddof=1)
    sd = np.where(sd > _FLOOR, sd, 1.0)
    z = (x - x.mean(axis=0)) / sd

    corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    vals, vecs = np.linalg.eigh((corr + corr.T) / 2.0)
    pc1 = z @ vecs[:, int(np.argmax(vals))]

    # Rolling standard deviation of the common component, trailing so a day is
    # classified on what had already happened by then.
    n = pc1.size
    roll = np.full(n, np.nan)
    for i in range(window - 1, n):
        roll[i] = pc1[i - window + 1:i + 1].std(ddof=1)

    finite = roll[np.isfinite(roll)]
    if finite.size == 0:
        return np.zeros(n, dtype=bool)
    cut = float(np.quantile(finite, quantile))
    return np.where(np.isfinite(roll), roll >= cut, False)


def equal_exposure_blocks(cov: np.ndarray, factors: list[str],
                          factor_block: list[str]) -> BlockAttribution:
    """The same split with every exposure set to one.

    A baseline that needs no security: it answers which blocks are intrinsically
    the noisiest and the most entangled, rather than which drive a particular
    holding. Useful next to a real security, because a block that is large here
    and small there is a block that security happens not to be exposed to, which
    is a different fact from the block being quiet.
    """
    return attribute(np.ones(len(factors)), cov, factors, factor_block)
