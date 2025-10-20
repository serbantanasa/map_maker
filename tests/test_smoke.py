from map_maker.generate import generate_world


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
