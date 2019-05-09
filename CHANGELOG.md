# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [unreleased]

## [1.0.3] 2019-05-09
### Fixed

- Fixed small memory leak in svd compression

### Added

- Optimized svd compression a bit


## [1.0.2] 2017-12-13

- Expand documentation to mpnum.sandwitch
- Restructured and added to online documentation


## [1.0.1] 2017-10-25
### Fixed

- Fix python2 compatibility issue in setup.py
- long_description in text format instead of RST


## [1.0.0] 2017-10-25
### Added

- Add `utils.extmath.randomized_svd` providing a fast approximate truncated SVD

### Changed
- `MPArray.compress`: `method='var'` requires `num_sweeps`
- Rename `MPArray.bdims` -> `MPArray.ranks`
- Rename `MPArray.pdims` -> `MPArray.shape`
- Rename `MPArray.plegs` -> `MPArray.ndims`
- Rename `MPArray.normal_form` -> `MPArray.canonical_form`
- Rename `MPArray.normalize` -> `MPArray.canonicalize`
- Rename `MPArray.get_phys` -> `MPArray.get`
- Rename `MPArray.paxis_iter` -> `MPArray.axis_iter`
- Rename `MPArray.pleg2bleg` -> `MPArray.leg2vleg`
- Rename `MPArray.dims` -> `MPArray.lt.shape`
- Rename `LocalTensors.normal_form` -> `LocalTensors.canonical_form`
- Rename `mparray.full_bdim` -> `mparray.full_rank`
- Rename `mparray.outer` -> `mparray.chain`
- Rename `mparray.louter` -> `mparray.localouter`
- Rename `mparray.linalg.mineig` -> `mparray.linalg.eig`
- Rename `mparray.linalg.mineig_sum` -> `mparray.linalg.eig_sum`
- Rename `mparray.tools` -> `mparray.utils`
- Rename `mparray.tools.check_nonneg_trunc` -> `mparray.utils.pmf.project_nonneg`
- Rename `mparray.tools.check_pmf` -> `mparray.utils.pmf.project_pmf`
- Rename `mparray.linalg.mineig` -> `mparray.linalg.eig`
- `mparray.linalg.eig`:
  - Rename parameter `max_num_sweeps` -> `num_sweeps` (now required)
  - Rename parameter `minimize_sites` -> `var_sites`
  - Change order of parameters
  - Require `num_sweeps` and one of `startvec` and `startvec_rank`

### Removed
- Remove `MPArray.bdim`, use `max(mpa.ranks)` instead
- Remove `mpo_to_global`, use `to_array_global` instead
- Remove `mpnum.tools.verify_real_nonnegative`, use `mpnum.utils.pmf.project_nonneg` instead


### Deprecated

- Deprecate support for Python 3.3
