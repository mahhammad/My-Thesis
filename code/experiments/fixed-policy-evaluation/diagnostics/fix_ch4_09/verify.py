"""Read-only audit of the persisted FIX09 replay, without new Monte Carlo."""
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
P=HERE/"results"/"controlled_uniform_replay.json"

def check():
    d=json.loads(P.read_text(encoding="utf8"))
    assert d["schema"]=="FIX-CH4-09-controlled-uniform-replay-v1"
    assert d["retrain"] is False and d["historical_test_loaded"] is False
    assert d["data"]["E08_fingerprint_match_by_split"]["train_labels"]
    assert d["data"]["matches_existing_e08_train_validation_fingerprints"] == all(
        d["data"]["E08_fingerprint_match_by_split"].values())
    assert d["reference"]["E05_exact_predictions_match_per_sample"]
    assert d["reference"]["E05_exact_center_max_abs_error"]<=1e-10
    assert d["policy"]["MC_walks_per_origin"]==300
    assert d["policy"]["MC_seed"]==20261103
    assert set(d["cases"])=={
        "mc_val_mc_centers","mc_val_exact_centers",
        "exact_val_mc_centers","exact_val_exact_centers"}
    assert abs(d["cases"]["exact_val_exact_centers"]["balanced_accuracy"] -
               .9676870748299321)<1e-12
    y_count=np.array([13,49,28])
    for key,c in d["cases"].items():
        conf=np.asarray(c["confusion_true_rows_pred_cols"])
        assert conf.shape==(3,3)
        np.testing.assert_array_equal(conf.sum(axis=1),y_count)
        assert len(c["predictions"])==90
        ba=np.mean(np.diag(conf)/y_count)
        assert abs(ba-c["balanced_accuracy"])<=1e-12,key
    for k,v in d["disagreement_rows"].items():
        assert len(v)==d["disagreement_counts"][k]
        assert all(0<=x["validation_position"]<90 for x in v)
    # Original stored data are NOT committed: never assert byte-identity.
    assert d["data"]["original_historical_generated_data_byte_identity_verifiable"] is False

    new=HERE/"results"/"mc_convergence.json"
    if new.is_file():
        e=json.loads(new.read_text(encoding="utf8"))
        assert e["schema"]=="FIX-CH4-09-MC-convergence-v1"
        assert e["historical_test_accessed"] is False and e["research_new_training"] is False
        assert e["budgets_are_nested"] is False
        assert e["rows"][0]["walks"]==300
        assert abs(e["rows"][0]["validation_BA"]-d["cases"]["mc_val_mc_centers"]["balanced_accuracy"])<1e-12
        assert [z["walks"] for z in e["rows"]]==[300,1200,4800,9600]
        for z in e["rows"][1:]:
            assert len(z["validation_predictions"])==90
            assert len(z["MC_train_representations"])==300
            assert len(z["MC_validation_representations"])==90
            assert z["validation_disagreement_count_vs_exact"]==len(z["disagreement_rows"])
            assert 0<=z["validation_BA"]<=1
            assert z["validation_TV_mean"]>=0
        print("PASS: independent higher-budget convergence output verified without rerunning MC.")
    print("PASS: FIX09 stored results consistent, historical files untouched, no rerun.")
    print("MC replay matches published two-decimal BA:",
          d["replayed_MC_BA_equals_historical_to_two_decimals"])
    print("Crossed BAs:",{k:round(v["balanced_accuracy"]*100,3)
                         for k,v in d["cases"].items()})

if __name__=="__main__":
    check()
