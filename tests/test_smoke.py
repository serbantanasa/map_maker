import json
from pathlib import Path

from map_maker.generate import export_dataset, generate_world


def test_determinism_small():
    w1 = generate_world(128, 96, 123, {})
    w2 = generate_world(128, 96, 123, {})
    # basic invariants
    assert w1["elev"].shape == (96, 128)
    assert w1["stats"]["land_ratio"] == w2["stats"]["land_ratio"]
    assert (w1["biome"] == w2["biome"]).all()


def test_all_land_biomes_colored():
    world = generate_world(128, 96, 321, {})
    land = world["elev"] >= world["params"]["ocean_level"]
    assert not (world["biome"][land] == 0).any()


def test_basin_labeling_and_stream_orders():
    world = generate_world(96, 64, 555, {})
    ocean = world["elev"] < world["params"]["ocean_level"]
    assert (world["basin_id"][ocean] == 0).all()
    assert world["strahler"].min() >= 1
    assert world["shreve"].min() >= 1
    rivers = world["rivers"]
    assert set(rivers.keys()) == {
        "geoms",
        "basin_id",
        "order_strahler",
        "order_shreve",
        "discharge",
        "length_km",
        "source_type",
        "mouth_type",
    }
    if rivers["geoms"]:
        length = len(rivers["geoms"])
        assert all(len(v) == length for k, v in rivers.items() if k != "geoms")


def test_export_dataset(tmp_path: Path):
    world = generate_world(64, 48, 777, {})
    out_dir = tmp_path / "dataset"
    export_dataset(out_dir, 777, world)
    manifest_path = out_dir / "dataset.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["seed"] == 777
    assert (out_dir / manifest["files"]["rasters"]["elev"]).exists()
