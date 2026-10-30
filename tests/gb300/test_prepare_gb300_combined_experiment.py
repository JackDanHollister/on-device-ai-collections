import json
from pathlib import Path
import sys

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "gb300"))
import prepare_gb300_combined_experiment as prep


def test_blind_union_maps_both_lanes_and_excludes_only_zero_pair(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with duckdb.connect() as db:
        db.sql("""SELECT * FROM (VALUES
            ('1',' Monks  Wood ','GB','England',NULL,NULL),
            ('2','monks wood','gb','england',52.4,-0.24),
            ('3','Greenwich','GB','England',51.4,0.0),
            ('4','Null Island','GB','England',0.0,0.0),
            ('5','Bad pair','GB','England',91.0,2.0),
            ('6','   ','GB','England',50.0,2.0),
            ('7','Monks Wood','GB','Scotland',NULL,NULL))
            t(gbifid,locality,countrycode,stateprovince,decimallatitude,decimallongitude)
            """).write_parquet(str(source / "source.parquet"))
    prep.integrity.save(source / "COMPLETE.json", {"status": "complete", "compacted_source_sha256": prep.integrity.sha(source / "source.parquet")})
    out = tmp_path / "out"
    manifest = prep.prepare(source, out)
    assert manifest["counts"]["records_by_cohort"] == {"production": 3, "evaluation": 2}
    assert manifest["counts"]["unique_candidates"] == 4
    assert manifest["counts"]["shared_unique_inputs"] == 1
    assert manifest["counts"]["excluded_zero_zero_with_locality"] == 1
    with duckdb.connect() as db:
        candidates = db.sql("SELECT * FROM read_parquet(?)", params=[str(out / "candidates.parquet")])
        assert set(candidates.columns) == {"candidate_id", "locality", "country", "occurrence_count", "example_identifier", "context_json"}
        assert all(set(json.loads(r[0])) == {"stateprovince"} for r in candidates.project("context_json").fetchall())
        refs = db.sql("SELECT * FROM read_parquet(?)", params=[str(out / "evaluation_only/references.parquet")])
        assert set(r[0] for r in refs.project("gbifid").fetchall()) == {"2", "3"}
    assert prep.hpc._preparation_identity(out / "candidates.parquet", manifest["artifacts"]["candidates"]["sha256"])["policy"]["no_reference_coordinates_or_memory_in_inference"]
    with pytest.raises(ValueError, match="fresh"):
        prep.prepare(source, out)
    prep.integrity.save(source / "COMPLETE.json", {"status": "complete", "compacted_source_sha256": "wrong"})
    with pytest.raises(ValueError, match="hash"):
        prep.prepare(source, tmp_path / "bad")
