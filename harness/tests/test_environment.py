"""Environment fingerprint: reading it, and what comparing two of them means.

Issue #107. Comparable-view equality (models, chunking, index-time flags)
does not capture stack drift, so an August high-water entry can hold passes
today's MinerU/jina-clip cannot reach, and every criterion-5 campaign ends
``rejected_regression`` on passes nobody can earn back.

The tests here fix the two decisions that make the fingerprint honest rather
than merely present:

1. **An unreadable environment is ``None``, never a dict of ``None``s.** Two
   fingerprints that know nothing would otherwise compare equal and claim a
   verified comparability neither has.
2. **Unknown is not the same as different.** Ledger entries written before
   this field existed carry no fingerprint at all, and declaring them
   incomparable would disarm recovery mode for the whole existing ledger --
   solving #107 by discarding the evidence #92 was built on.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import environment  # noqa: E402


def _make_site_packages(tmp_path: Path, name: str, packages: dict) -> Path:
    """Build a fake venv whose dist-info directories carry ``packages``."""
    site = tmp_path / name / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    for dist_name, version in packages.items():
        (site / f"{dist_name}-{version}.dist-info").mkdir()
    return site


class TestReadingVersions:
    def test_reads_the_version_out_of_a_dist_info_directory_name(self, tmp_path):
        site = _make_site_packages(tmp_path, "venv", {"transformers": "4.57.6"})
        assert environment.read_stack_versions([site])["transformers"] == "4.57.6"

    def test_normalizes_the_hyphen_underscore_difference(self, tmp_path):
        """`pip install sentence-transformers` writes `sentence_transformers-*.dist-info`."""
        site = _make_site_packages(
            tmp_path, "venv", {"sentence_transformers": "5.6.1", "faiss_cpu": "1.15.0"}
        )
        versions = environment.read_stack_versions([site])
        assert versions["sentence-transformers"] == "5.6.1"
        assert versions["faiss-cpu"] == "1.15.0"

    def test_a_package_that_is_not_installed_reads_as_none(self, tmp_path):
        site = _make_site_packages(tmp_path, "venv", {"transformers": "4.57.6"})
        assert environment.read_stack_versions([site])["mineru"] is None

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        versions = environment.read_stack_versions([tmp_path / "nope"])
        assert set(versions) == set(environment.TRACKED_PACKAGES)
        assert all(value is None for value in versions.values())

    def test_the_first_root_holding_a_package_wins(self, tmp_path):
        """The isolated venv is searched first: it is the one that builds the index."""
        isolated = _make_site_packages(tmp_path, "isolated", {"transformers": "4.57.6"})
        product = _make_site_packages(tmp_path, "product", {"transformers": "5.16.1"})
        assert environment.read_stack_versions([isolated, product])["transformers"] == "4.57.6"
        assert environment.read_stack_versions([product, isolated])["transformers"] == "5.16.1"


class TestFingerprint:
    def test_an_environment_with_nothing_readable_fingerprints_as_none(self, tmp_path):
        """Not a dict of Nones -- see this module's docstring, decision 1."""
        environments = {environment.PRODUCT_ENV: [tmp_path / "nope"]}
        assert environment.environment_fingerprint(environments) is None

    def test_a_readable_environment_fingerprints_as_a_dict(self, tmp_path):
        site = _make_site_packages(tmp_path, "venv", {"transformers": "4.57.6"})
        fingerprint = environment.environment_fingerprint({environment.PRODUCT_ENV: [site]})
        assert fingerprint["packages"]["product:transformers"] == "4.57.6"
        assert fingerprint["schema"] == environment.FINGERPRINT_SCHEMA

    def test_the_two_environments_are_recorded_separately(self, tmp_path):
        """transformers 4.x isolated and 5.x product mean different things.

        One merged key would record whichever was searched first and silently
        drop the other -- and the dropped one is a real input to the result.
        """
        isolated = _make_site_packages(tmp_path, "isolated", {"transformers": "4.57.6"})
        product = _make_site_packages(tmp_path, "product", {"transformers": "5.16.1"})
        fingerprint = environment.environment_fingerprint(
            {environment.ISOLATED_ENV: [isolated], environment.PRODUCT_ENV: [product]}
        )
        assert fingerprint["packages"]["isolated:transformers"] == "4.57.6"
        assert fingerprint["packages"]["product:transformers"] == "5.16.1"

    def test_the_fingerprint_survives_a_json_round_trip(self, tmp_path):
        """It is written into a ledger entry, so it has to be plain JSON."""
        site = _make_site_packages(tmp_path, "venv", {"mineru": "2.6.3"})
        fingerprint = environment.environment_fingerprint({environment.ISOLATED_ENV: [site]})
        assert json.loads(json.dumps(fingerprint)) == fingerprint

    def test_the_real_environment_never_raises(self):
        """Whatever this machine has installed, reading it is best-effort."""
        environment.environment_fingerprint()

    def test_the_default_environments_name_both_and_only_both(self):
        assert set(environment.default_environments()) == {
            environment.ISOLATED_ENV,
            environment.PRODUCT_ENV,
        }


class TestComparison:
    def test_identical_stacks_match(self):
        one = {"schema": 1, "packages": {"isolated:mineru": "2.6.3", "isolated:torch": "2.6.0"}}
        assert environment.compare(one, dict(one)) == environment.MATCH

    def test_a_changed_version_differs(self):
        old = {"schema": 1, "packages": {"isolated:mineru": "2.6.3"}}
        new = {"schema": 1, "packages": {"isolated:mineru": "2.7.0"}}
        assert environment.compare(old, new) == environment.DIFFERS

    @pytest.mark.parametrize(
        "pair",
        [
            (None, {"schema": 1, "packages": {"isolated:mineru": "2.6.3"}}),
            ({"schema": 1, "packages": {"isolated:mineru": "2.6.3"}}, None),
            (None, None),
        ],
    )
    def test_a_missing_fingerprint_is_unknown_not_different(self, pair):
        """Decision 2: pre-#107 ledger entries stay usable, flagged as unverified."""
        assert environment.compare(*pair) == environment.UNKNOWN

    def test_two_stacks_sharing_no_known_package_are_unknown(self):
        """Nothing was actually compared, so nothing was actually verified."""
        one = {"schema": 1, "packages": {"isolated:mineru": "2.6.3", "isolated:torch": None}}
        two = {"schema": 1, "packages": {"isolated:mineru": None, "isolated:torch": "2.6.0"}}
        assert environment.compare(one, two) == environment.UNKNOWN

    def test_packages_only_one_side_knows_are_skipped_not_counted_as_difference(self):
        one = {"schema": 1, "packages": {"isolated:mineru": "2.6.3", "isolated:torch": "2.6.0"}}
        two = {"schema": 1, "packages": {"isolated:mineru": "2.6.3", "isolated:torch": None}}
        assert environment.compare(one, two) == environment.MATCH

    def test_a_schema_change_is_unknown_rather_than_a_silent_mismatch(self):
        """A later schema may track different packages; comparing across is a guess."""
        one = {"schema": 1, "packages": {"isolated:mineru": "2.6.3"}}
        two = {"schema": 2, "packages": {"isolated:mineru": "2.6.3"}}
        assert environment.compare(one, two) == environment.UNKNOWN

    def test_the_difference_is_described_field_by_field(self):
        old = {"schema": 1, "packages": {"isolated:mineru": "2.6.3", "isolated:torch": "2.6.0"}}
        new = {"schema": 1, "packages": {"isolated:mineru": "2.7.0", "isolated:torch": "2.6.0"}}
        described = environment.describe_difference(old, new)
        assert described == ["isolated:mineru: this launch '2.7.0', ledger '2.6.3'"]

    def test_describing_a_match_yields_nothing(self):
        one = {"schema": 1, "packages": {"isolated:mineru": "2.6.3"}}
        assert environment.describe_difference(one, dict(one)) == []
