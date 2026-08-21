# Citation audit

Audit date: 2026-08-18.

## Integrity checks

- Every one of the 28 citation keys used in main.tex has exactly one entry in refs.bib.
- refs.bib contains no unused entries, duplicate keys, placeholder citations, or unresolved author-year stubs.
- Recent preprints were checked against their primary arXiv records; proceedings metadata was checked against the publisher or conference record where available.

## Claim-to-source corrections

| Claim in the paper | Audit result and action |
|---|---|
| CIO calibrates an uncertainty set that intersects the next inverse-feasible set. | Supported by Lin, Delage, and Chan (2024), including its finite-sample coverage theorem. Retained. |
| CIO provides a diameter or Hausdorff-width guarantee for the full inverse-feasible set. | Not supported by the CIO paper. The old attribution was removed. The normalized inverse-fiber width assumption is now explicitly presented as CP-IRL's additional identification condition. |
| Data-driven and predict-then-calibrate robust optimization solve the same latent inverse problem as CIO. | Too broad. The related-work text now distinguishes uncertainty around observed or predicted objective coefficients from CIO's latent inverse-feasible-set setting. |
| Existing feasible-reward-set IRL papers all use the same sample model or PAC assumptions. | Not supported as a blanket statement. Replaced by a narrower statement that the papers use different observation models and assumptions; the generative-model and multiple-expert claim is attributed specifically to Poiani et al. (2024). |
| Rectangularity and Bayesian ambiguity are the same robust-MDP device. | Incorrect conflation. Rectangularity is cited to Iyengar, Nilim and El Ghaoui, and Wiesemann et al.; Petrik and Russel is cited separately for Bayesian ambiguity sets. |
| EPIC removes reward transformations, DARD restricts comparisons to realizable transitions, and STARC gives a general regret-sensitive framework. | Supported by the respective primary conference records. Retained with narrower wording. |
| No prior paper conformally calibrates inverse-feasible reward sets from an IRL demonstrator population. | A negative literature claim cannot be established exhaustively. The manuscript now says “we are not aware of prior work” and limits the novelty claim to the stated calibration unit and setting. |

## Metadata corrections

- Added the official NeurIPS volume, page range, DOI, and paper URL for Conformal Inverse Optimization.
- Corrected the third author of Predict-then-Calibrate to Xiaocheng Li.
- Updated the Management Science publication year for Chan, Lee, and Terekhov to the journal issue year, 2019.
- Replaced the outdated Skalse citation key with the official ICML 2023 record and full author metadata.
- Added official NeurIPS metadata for Chenreddy et al., Poiani et al., and Direct Preference Optimization where available.
- Added the DOI for Nilim and El Ghaoui and explicit arXiv identifiers for recent preprints.

This report records bibliographic and attribution checks; it is not a claim that the literature search proves absence of all related unpublished work.
